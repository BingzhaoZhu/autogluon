import copy
import logging
from typing import Callable, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from torch.nn.modules.loss import _Loss
from torchmetrics.aggregation import BaseAggregator

from ..constants import AUTOMM, LM_TARGET, LOGITS, T_FEW, TEMPLATE_LOGITS, WEIGHT
from ..data.mixup import MixupModule, multimodel_mixup
from .utils import apply_layerwise_lr_decay, apply_single_lr, apply_two_stages_lr, get_lr_scheduler, get_optimizer

logger = logging.getLogger(AUTOMM)

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXent(nn.Module):
    def __init__(self, temperature=1.0):
        """NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
        Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation

        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch

        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size = z_i.size(0)

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float().to(z_i.get_device())
        numerator = torch.exp(positives / self.temperature)

        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss


class ContrastiveTransformations:
    def __init__(self, model, mode, problem_type, corruption_rate):
        self.model = model
        self.mode = mode if mode is not None else "identical"
        self.problem_type = problem_type
        self.corruption_rate = corruption_rate

    def __call__(self, batch):
        if self.mode == "identical":
            return self.identical(batch)
        elif self.mode == "random_perm":
            return self.random_perm(batch)
        elif self.mode == "random_block":
            return self.random_block(batch)
        else:
            raise ValueError(
                f"Current mode {self.mode} is not supported."
                "Consider choosing from the following options:"
                "identical, random_perm."
            )

    def identical(self, batch):
        batch = copy.deepcopy(batch)
        return batch

    def random_block(self, batch):
        corruption_rate = self.corruption_rate
        batch = copy.deepcopy(batch)
        for permodel in self.model.model:
            if hasattr(permodel, "categorical_key"):
                categorical_features = []
                for categorical_feature in batch[permodel.categorical_key]:
                    if torch.rand(1) < corruption_rate:
                        values, indices = torch.mode(categorical_feature, 0)
                        categorical_feature[:] = values
                    categorical_features.append(categorical_feature)
                batch[permodel.categorical_key] = tuple(categorical_features)
            if hasattr(permodel, "numerical_key"):
                numerical_features = batch[permodel.numerical_key]
                _, m = numerical_features.size()
                for i in range(m):
                    if torch.rand(1) < corruption_rate:
                        values, indices = torch.mode(numerical_features[:, i], 0)
                        numerical_features[:, i] = values
                batch[permodel.numerical_key] = numerical_features
        return batch

    def random_perm(self, batch):
        corruption_rate = self.corruption_rate
        batch = copy.deepcopy(batch)
        batch_size, = batch[self.model.label_key].size()
        corruption_len = int(batch_size * corruption_rate)
        for permodel in self.model.model:
            if hasattr(permodel, "categorical_key"):
                categorical_features = []
                for categorical_feature in batch[permodel.categorical_key]:
                    random_idx = torch.randint(high=batch_size, size=(batch_size,))
                    random_sample = categorical_feature[random_idx]
                    corruption_idx = torch.randperm(batch_size)[:corruption_len]
                    corruption_mask = torch.zeros_like(categorical_feature, dtype=torch.bool)
                    corruption_mask[corruption_idx] = True
                    positive = torch.where(corruption_mask, random_sample, categorical_feature)
                    categorical_features.append(positive)
                batch[permodel.categorical_key] = tuple(categorical_features)
            if hasattr(permodel, "numerical_key"):
                numerical_features = batch[permodel.numerical_key]
                _, m = numerical_features.size()
                indices = torch.argsort(torch.rand(*numerical_features.shape), dim=0)
                random_sample = numerical_features[indices, torch.arange(m).unsqueeze(0)]
                corruption_mask = torch.zeros_like(numerical_features, dtype=torch.bool)
                for i in range(m):
                    corruption_idx = torch.randperm(batch_size)[:corruption_len]
                    corruption_mask[corruption_idx, i] = True
                batch[permodel.numerical_key] = torch.where(corruption_mask, random_sample, numerical_features)

        return batch


class PretrainerLitModule(pl.LightningModule):
    """
    Control the loops for training, evaluation, and prediction. This module is independent of
    the model definition. This class inherits from the Pytorch Lightning's LightningModule:
    https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: nn.Module,
        optim_type: Optional[str] = None,
        lr_choice: Optional[str] = None,
        lr_schedule: Optional[str] = None,
        lr: Optional[float] = None,
        lr_decay: Optional[float] = None,
        end_lr: Optional[Union[float, int]] = None,
        lr_mult: Optional[Union[float, int]] = None,
        weight_decay: Optional[float] = None,
        warmup_steps: Optional[int] = None,
        loss_func: Optional[_Loss] = None,
        validation_metric: Optional[torchmetrics.Metric] = None,
        validation_metric_name: Optional[str] = None,
        custom_metric_func: Callable = None,
        test_metric: Optional[torchmetrics.Metric] = None,
        efficient_finetune: Optional[str] = None,
        trainable_param_names: Optional[List[str]] = None,
        mixup_fn: Optional[MixupModule] = None,
        mixup_off_epoch: Optional[int] = 0,
        problem_type: Optional[str] = None,
        augmentation_mode: Optional[str] = None,
        corruption_rate: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        model
            A Pytorch model
        optim_type
            Optimizer type. We now support:
            - adamw
            - adam
            - sgd
        lr_choice
            How to set each layer's learning rate. If not specified, the default is a single
            learnng rate for all layers. Otherwise, we now support two choices:
            - two_stages
                The layers in the pretrained models have a small learning rate (lr * lr_mult),
                while the newly added head layers use the provided learning rate.
            - layerwise_decay
                The layers have decreasing learning rate from the output end to the input end.
                The intuition is that later layers are more task-related, hence larger learning rates.
        lr_schedule
            Learning rate schedule. We now support:
            - cosine_decay
                Linear warmup followed by cosine decay
            - polynomial_decay
                Linear warmup followed by polynomial decay
        lr
            Learning rate.
        lr_decay
            The learning rate decay factor (0, 1). It is used only when lr_choice is "layerwise_decay".
        end_lr
            The final learning rate after decay.
        lr_mult
            The learning rate multiplier (0, 1). It is used only when lr_choice is "two_stages".
        weight_decay
            The weight decay to regularize layer weights' l2 norm.
        warmup_steps
            How many steps to warmup learning rate. If a float (0, 1), it would represent the
            percentage of steps over all the training steps. The actual number is calculated as
            "int(warmup_steps * max_steps)". If an integer, it would be the exact step number.
        loss_func
            A Pytorch loss module, e.g., nn.CrossEntropyLoss().
        validation_metric
            A torchmetrics module used in the validation stage, e.g., torchmetrics.Accuracy().
        validation_metric_name
            Name of validation metric in case that validation_metric is a aggregation metric,
            e.g., torchmetrics.MeanMetric, whose name can't reflect the real metric name.
        custom_metric_func
            A customized metric function in case that torchmetrics doesn't have the metric.
            It is generally used together with torchmetrics' aggregators, e.g., torchmetrics.MeanMetric.
            Refer to https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/aggregation.py
        test_metric
            A torchmetrics module used in the test stage, e.g., torchmetrics.Accuracy().
        efficient_finetune
            Whether to use efficient finetuning strategies. This will be helpful for fast finetuning of large backbones.
            We support options such as:

            - bit_fit (only finetune the bias terms)
            - norm_fit (only finetune the weights in norm layers / bias layer)
            - lora, lora_bias, lora_norm (only finetunes decomposition matrices inserted into model, in combination with either bit_fit or norm_fit)
            - ia3, ia3_bias, ia3_norm (adds vector that scales activations by learned vectors, in combination with either bit_fit or norm_fit)
            - None (do not use efficient finetuning strategies)

        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "validation_metric", "test_metric", "loss_func"])
        self.model = model
        self.loss_func = NTXent(temperature=1)
        self.contrastive_fn = ContrastiveTransformations(model,
                                                         mode=augmentation_mode,
                                                         problem_type=problem_type,
                                                         corruption_rate=corruption_rate,
                                                         )

    def _compute_loss(
        self,
        output: Dict,
        positive: Dict,
    ):
        loss = 0
        for per_key, _ in output.items():
            per_output = output[per_key]
            per_positive = positive[per_key]
            weight = per_output[WEIGHT] if WEIGHT in per_output else 1
            loss += (
                self.loss_func(
                    z_i=per_output[LOGITS].squeeze(dim=1),
                    z_j=per_positive[LOGITS].squeeze(dim=1),
                )
                * weight
            )
        return loss

    def _shared_step(
        self,
        batch: Dict,
    ):
        corrupted_batch = self.contrastive_fn(batch)
        output = self.model(batch)
        positive = self.model(corrupted_batch)
        loss = self._compute_loss(output=output, positive=positive)
        return output, loss

    def training_step(self, batch, batch_idx):
        """
        Per training step. This function is registered by pl.LightningModule.
        Refer to https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#training-loop

        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data, including both input data and
            ground-truth labels. The mini-batch data are passed to each individual model,
            which indexes its required input data by keys with its model prefix. The
            ground-truth labels are used here to compute the training loss.
        batch_idx
            Index of mini-batch.

        Returns
        -------
        Average loss of the mini-batch data.
        """
        output, loss = self._shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Per validation step. This function is registered by pl.LightningModule.
        Refer to https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#validation

        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data, including both input data and
            ground-truth labels. The mini-batch data are passed to each individual model,
            which indexes its required input data by keys with its model prefix. The
            ground-truth labels are used here to compute the validation loss and metric.
            The validation metric is used for top k model selection and early stopping.
        batch_idx
            Index of mini-batch.
        """
        output, loss = self._shared_step(batch)
        self.log("val_loss", loss)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Per prediction step. This function is registered by pl.LightningModule.
        Refer to https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#prediction-loop

        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data.
            The mini-batch data are passed to each individual model,
            which indexes its required input data by keys with its model prefix.
            Ground-truth labels are not needed for prediction.
        batch_idx
            Index of mini-batch.
        dataloader_idx
            Index of dataloader.
        Returns
        -------
        A dictionary with the mini-batch's logits and features.
        """
        output = self.model(batch)
        return output[self.model.prefix]

    def configure_optimizers(self):
        """
        Configure optimizer. This function is registered by pl.LightningModule.
        Refer to https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        Returns
        -------
        [optimizer]
            Optimizer.
        [sched]
            Learning rate scheduler.
        """
        kwargs = dict(
            model=self.model,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        logger.debug("applying single learning rate...")
        grouped_parameters = apply_single_lr(
            **kwargs,
        )

        optimizer = get_optimizer(
            optim_type=self.hparams.optim_type,
            optimizer_grouped_parameters=grouped_parameters,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        logger.debug(f"trainer.max_steps: {self.trainer.max_steps}")
        if self.trainer.max_steps is None or -1:
            max_steps = (
                len(self.trainer.datamodule.train_dataloader())
                * self.trainer.max_epochs
                // self.trainer.accumulate_grad_batches
            )
            logger.debug(
                f"len(trainer.datamodule.train_dataloader()): {len(self.trainer.datamodule.train_dataloader())}"
            )
            logger.debug(f"trainer.max_epochs: {self.trainer.max_epochs}")
            logger.debug(f"trainer.accumulate_grad_batches: {self.trainer.accumulate_grad_batches}")
        else:
            max_steps = self.trainer.max_steps

        logger.debug(f"max steps: {max_steps}")

        warmup_steps = self.hparams.warmup_steps
        if isinstance(warmup_steps, float):
            warmup_steps = int(max_steps * warmup_steps)

        logger.debug(f"warmup steps: {warmup_steps}")
        logger.debug(f"lr_schedule: {self.hparams.lr_schedule}")
        scheduler = get_lr_scheduler(
            optimizer=optimizer,
            num_max_steps=max_steps,
            num_warmup_steps=warmup_steps,
            lr_schedule=self.hparams.lr_schedule,
            end_lr=self.hparams.end_lr,
        )

        sched = {"scheduler": scheduler, "interval": "step"}
        logger.debug("done configuring optimizer and scheduler")
        return [optimizer], [sched]

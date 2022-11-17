import logging
from typing import Callable, Dict, List, Optional, Union
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from torch.nn.modules.loss import _Loss
from torchmetrics.aggregation import BaseAggregator
import learn2learn as l2l

from ..constants import AUTOMM, LM_TARGET, LOGITS, T_FEW, TEMPLATE_LOGITS, WEIGHT
from ..data.mixup import MixupModule, multimodel_mixup
from .utils import apply_layerwise_lr_decay, apply_single_lr, apply_two_stages_lr, get_lr_scheduler, get_optimizer

logger = logging.getLogger(AUTOMM)


class MetaModule(pl.LightningModule):
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
        trainable_param_names: Optional[List] = None,
        mixup_fn: Optional[MixupModule] = None,
        mixup_off_epoch: Optional[int] = 0,
        model_postprocess_fn: Callable = None,
        all_pretrain_tasks=None,
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
        self.save_hyperparameters(
            ignore=["model", "validation_metric", "test_metric", "loss_func", "model_postprocess_fn", "all_pretrain_tasks"]
        )
        self.model = model
        self.validation_metric = validation_metric
        self.validation_metric_name = f"val_{validation_metric_name}"
        self.loss_func = loss_func
        self.mixup_fn = mixup_fn
        if isinstance(validation_metric, BaseAggregator) and custom_metric_func is None:
            raise ValueError(
                f"validation_metric {validation_metric} is an aggregation metric,"
                "which must be used with a customized metric function."
            )
        self.custom_metric_func = custom_metric_func
        self.model_postprocess_fn = model_postprocess_fn
        self.trainable_param_names = trainable_param_names if trainable_param_names else []
        # self.freeze_backbone(freeze=True)
        self.all_pretrain_tasks = all_pretrain_tasks
        self.adaptation_steps = 5

    def _compute_loss(
        self,
        output: Dict,
        label: torch.Tensor,
    ):
        loss = 0
        for _, per_output in output.items():
            weight = per_output[WEIGHT] if WEIGHT in per_output else 1
            if (
                TEMPLATE_LOGITS in per_output and self.model.prefix == T_FEW
            ):  # Do only add template loss if T-Few. #TODO Add compatibility to Fusion models.
                loss += self._compute_template_loss(per_output, label) * weight
            else:
                loss += (
                    self.loss_func(
                        input=per_output[LOGITS].squeeze(dim=1),
                        target=label,
                    )
                    * weight
                )
        return loss

    def _compute_metric_score(
        self,
        metric: torchmetrics.Metric,
        custom_metric_func: Callable,
        logits: torch.Tensor,
        label: torch.Tensor,
    ):
        if isinstance(metric, (torchmetrics.AUROC, torchmetrics.AveragePrecision)):
            prob = F.softmax(logits.float(), dim=1)
            metric.update(preds=prob[:, 1], target=label)  # only for binary classification
        elif isinstance(metric, BaseAggregator):
            metric.update(custom_metric_func(logits, label))
        else:
            metric.update(logits.squeeze(dim=1), label)

    def _shared_step(
        self,
        batch: Dict,
        model=None,
    ):
        if model is None:
            model = self.model
        label = batch[self.model.label_key]
        if self.mixup_fn is not None:
            self.mixup_fn.mixup_enabled = self.training & (self.current_epoch < self.hparams.mixup_off_epoch)
            batch, label = multimodel_mixup(batch=batch, model=model, mixup_fn=self.mixup_fn)
        output = model(batch)
        loss = self._compute_loss(output=output, label=label)
        return output, loss

    def _set_loader_spec(self, loader):
        self.model = l2l.algorithms.MAML(loader["model"], lr=0.001, first_order=False, allow_unused=True).to("cuda:0")
        module = loader["data_module"]
        module.setup("fit")
        return module.train_dataloader()

    def _split_batch(self, batch, queries):
        N = batch[self.model.label_key].size(0)
        # print(np.unique(batch[self.model.label_key]))
        support, query = {}, {}

        # Separate data into adaptation and evaluation sets
        support_indices = np.zeros(N, dtype=bool)
        selection = np.random.permutation(N)[:int(N * queries)]
        support_indices[selection] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)

        for per_key in batch:
            if "categorical_transformer_categorical" in per_key:
                per_support, per_query = [], []
                for t in batch[per_key]:
                    per_support.append(t[support_indices].to("cuda:0"))
                    per_query.append(t[query_indices].to("cuda:0"))
                support[per_key] = tuple(per_support)
                query[per_key] = tuple(per_query)
            else:
                support[per_key] = batch[per_key][support_indices].to("cuda:0")
                query[per_key] = batch[per_key][query_indices].to("cuda:0")

        return support, query

    @torch.enable_grad()
    def meta_learn(self, meta_batch, queries=0.5):
        loss = 0
        for idx in meta_batch:
            loader = self.all_pretrain_tasks[idx]
            loader = self._set_loader_spec(loader)
            learner = self.model.clone()
            learner.train()

            batch = next(iter(loader))
            support, query = self._split_batch(batch, queries)

            # Adapt the model
            for step in range(self.adaptation_steps):
                _, train_error = self._shared_step(support, learner)
                learner.adapt(train_error)

            # Evaluating the adapted model
            _, query_loss = self._shared_step(query, learner)
            loss += query_loss
        return loss

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
        # if self.current_epoch >= 2:
        #     self.freeze_backbone(freeze=False)
        loss = self.meta_learn(batch)
        print(loss)
        self.log("train_loss", loss)
        return loss

    def freeze_backbone(self, freeze=True):
        for params in self.model.fusion_transformer.parameters():
            params.requires_grad = not freeze
        return


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
        loss = self.meta_learn(batch)
        # if self.model_postprocess_fn:
        #     output = self.model_postprocess_fn(output)
        # By default, on_step=False and on_epoch=True
        self.log("val_loss", loss)
        # self._compute_metric_score(
        #     metric=self.validation_metric,
        #     custom_metric_func=self.custom_metric_func,
        #     logits=output[self.model.prefix][LOGITS],
        #     label=batch[self.model.label_key],
        # ),
        self.log(
            self.validation_metric_name,
            self.validation_metric,
            on_step=False,
            on_epoch=True,
        )

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
        if self.model_postprocess_fn:
            output = self.model_postprocess_fn(output)

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
        if self.hparams.lr_choice == "two_stages":
            logger.debug("applying 2-stage learning rate...")
            grouped_parameters = apply_two_stages_lr(
                lr_mult=self.hparams.lr_mult,
                return_params=True,
                **kwargs,
            )
        elif self.hparams.lr_choice == "layerwise_decay":
            logger.debug("applying layerwise learning rate decay...")
            grouped_parameters = apply_layerwise_lr_decay(
                lr_decay=self.hparams.lr_decay,
                efficient_finetune=self.hparams.efficient_finetune,
                trainable_param_names=self.trainable_param_names,
                **kwargs,
            )
        else:
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
            if "deepspeed" in self.trainer.strategy.strategy_name:
                max_steps = 1
            else:
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

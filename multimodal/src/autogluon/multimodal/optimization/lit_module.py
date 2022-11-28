import copy
import logging
from typing import Callable, Dict, List, Optional, Union
import boto3
import os
import time
import json
import shutil
from pathlib import Path

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

class LitModule(pl.LightningModule):
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
        is_pretrain=None,
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
            ignore=["model", "validation_metric", "test_metric", "loss_func", "model_postprocess_fn"]
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
        self.is_pretrain = is_pretrain
        self.prev_state_dic = None
        self.current_iter = 0

    def _compute_template_loss(
        self,
        per_output: Dict,
        label: torch.Tensor,
    ):
        logits = per_output[TEMPLATE_LOGITS]
        choices_scores = per_output[LOGITS]
        lm_target = per_output[LM_TARGET]

        num_choices = self.model.num_classes
        bs = int(lm_target.size(0) / num_choices)

        lm_loss = F.cross_entropy(
            logits.view(bs, num_choices, *logits.size()[1:])[range(bs), label].flatten(0, 1),
            lm_target.view(bs, num_choices, -1)[range(bs), label].flatten(0, 1),
        )
        if self.model.mc_loss > 0:
            mc_loss = F.cross_entropy(choices_scores, label)
        else:
            mc_loss = 0.0

        if self.model.unlikely_loss > 0:
            cand_loglikely = -F.cross_entropy(logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none").view(
                bs, num_choices, -1
            )
            cand_loglikely += (lm_target < 0).view(bs, num_choices, -1) * -100
            cand_loglikely[range(bs), label] = -100
            unlikely_loss = -torch.log(1 - torch.exp(cand_loglikely) + 1e-2).sum() / (cand_loglikely != -100).sum()
        else:
            unlikely_loss = 0.0

        return lm_loss + mc_loss * self.model.mc_loss + unlikely_loss * self.model.unlikely_loss

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

    def _save_s3(
        self,
        target='ec2/2022_09_14/cross_table_pretrain/pretrained_hogwild.ckpt',
        save_diff=False,
    ):
        current_state_dic = self.model.fusion_transformer.state_dict()
        if self.prev_state_dic is None:
            self.prev_state_dic = copy.deepcopy(current_state_dic)
        if save_diff:
            with torch.no_grad():
                for name in current_state_dic:
                    current_state_dic[name] = current_state_dic[name]-self.prev_state_dic[name]

        while True:
            try:
                checkpoint = {
                    "state_dict": {name: param for name, param in
                                   current_state_dic.items()}
                }
                torch.save(checkpoint, os.path.join("./", "pretrained.ckpt"))
                s3 = boto3.resource('s3')
                s3.Bucket('automl-benchmark-bingzzhu').upload_file('./pretrained.ckpt', target)
                break
            except:
                pass
                # time.sleep(5)

    def _save_pretrained_ckpt(self):
        s3_client = boto3.client('s3')
        BUCKET = 'automl-benchmark-bingzzhu'
        PREFIX = 'ec2/2022_09_14/cross_table_pretrain/iter_' + str(self.current_iter)
        file_names, folders = self.get_file_folders(s3_client, BUCKET, PREFIX)
        self.download_files(
            s3_client,
            BUCKET,
            "./",
            file_names,
            folders
        )

        files = os.listdir('./ec2/2022_09_14/cross_table_pretrain/iter_' + str(self.current_iter))

        new_pretrained = copy.deepcopy(self.prev_state_dic)
        for file_name in files:
            ckpt_path = './ec2/2022_09_14/cross_table_pretrain/iter_' + str(self.current_iter) + '/' + file_name
            state_dict = torch.load(ckpt_path, map_location=torch.device("cuda"))["state_dict"]
            with torch.no_grad():
                for name in new_pretrained:
                    new_pretrained[name] = new_pretrained[name] + state_dict[name]
        self.model.fusion_transformer.load_state_dict(new_pretrained)
        self._save_s3('ec2/2022_09_14/cross_table_pretrain/iter_' + str(self.current_iter) + '/pretrained.ckpt')
        self.prev_state_dic = copy.deepcopy(new_pretrained)
        shutil.rmtree('./ec2/')

    def _shared_step(
        self,
        batch: Dict,
    ):
        label = batch[self.model.label_key]
        if self.mixup_fn is not None:
            self.mixup_fn.mixup_enabled = self.training & (self.current_epoch < self.hparams.mixup_off_epoch)
            batch, label = multimodel_mixup(batch=batch, model=self.model, mixup_fn=self.mixup_fn)
        output = self.model(batch)
        loss = self._compute_loss(output=output, label=label)
        return output, loss


    def _process_lock(self):
        keep_waiting = True
        while keep_waiting:
            try:
                print("saving ckpt to s3")
                self._save_s3(
                    'ec2/2022_09_14/cross_table_pretrain/iter_' + str(self.current_iter) + '/' + self.is_pretrain[
                        'name'] + '.ckpt', True)
                break
            except:
                pass

        while keep_waiting:
            bucket = 'automl-benchmark-bingzzhu'
            File = 'ec2/2022_09_14/cross_table_pretrain/iter_' + str(self.current_iter) + '/'
            objs = boto3.client('s3').list_objects_v2(Bucket=bucket, Prefix=File)
            num_waiting = objs['KeyCount']

            bucket = 'automl-benchmark-bingzzhu'
            File = 'ec2/2022_09_14/cross_table_pretrain/iter_' + str(-1) + '/'
            objs = boto3.client('s3').list_objects_v2(Bucket=bucket, Prefix=File)
            num_failed = objs['KeyCount']

            print("n_waiting:", num_waiting, "num_failed:", num_failed)

            if num_waiting + num_failed >= self.is_pretrain["num_tasks"]:
                break

        try:
            s3 = boto3.resource('s3')
            s3.Bucket('automl-benchmark-bingzzhu').download_file(
                'ec2/2022_09_14/cross_table_pretrain/iter_' + str(self.current_iter) + '/pretrained.ckpt',
                './pretrained_.ckpt'
            )
            pretrain_path = os.path.join("./", 'pretrained_.ckpt')
            state_dict = torch.load(pretrain_path, map_location=torch.device("cuda"))["state_dict"]
            self.model.fusion_transformer.load_state_dict(state_dict)
            self.prev_state_dic = copy.deepcopy(state_dict)
        except:
            self._save_pretrained_ckpt()
            torch.cuda.empty_cache()

        return

    def get_file_folders(self, s3_client, bucket_name, prefix=""):
        file_names = []
        folders = []

        default_kwargs = {
            "Bucket": bucket_name,
            "Prefix": prefix
        }
        next_token = ""

        while next_token is not None:
            updated_kwargs = default_kwargs.copy()
            if next_token != "":
                updated_kwargs["ContinuationToken"] = next_token

            response = s3_client.list_objects_v2(**updated_kwargs)
            contents = response.get("Contents")

            for result in contents:
                key = result.get("Key")
                if key[-1] == "/":
                    folders.append(key)
                else:
                    file_names.append(key)

            next_token = response.get("NextContinuationToken")

        return file_names, folders

    def download_files(self, s3_client, bucket_name, local_path, file_names, folders):
        local_path = Path(local_path)

        for folder in folders:
            folder_path = Path.joinpath(local_path, folder)
            folder_path.mkdir(parents=True, exist_ok=True)

        for file_name in file_names:
            file_path = Path.joinpath(local_path, file_name)

            file_path.parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(
                bucket_name,
                file_name,
                str(file_path)
            )

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
        if self.is_pretrain["is_pretrain"]:
            if self.current_iter % self.is_pretrain["upload_per_n_iter"] == 0:
                self._process_lock()

            if self.current_iter % self.is_pretrain["iter_per_save"] == 0:
                self._save_s3(target='ec2/2022_09_14/cross_table_pretrain/raw_hog/pretrained_' + str(self.current_iter) + '.ckpt')

            if self.current_iter >= self.is_pretrain["max_iter"]:
                raise Exception("Pretraining reached max iteration")

            self.current_iter += 1

        output, loss = self._shared_step(batch)
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
        output, loss = self._shared_step(batch)
        if self.model_postprocess_fn:
            output = self.model_postprocess_fn(output)
        # By default, on_step=False and on_epoch=True
        self.log("val_loss", loss)
        self._compute_metric_score(
            metric=self.validation_metric,
            custom_metric_func=self.custom_metric_func,
            logits=output[self.model.prefix][LOGITS],
            label=batch[self.model.label_key],
        ),
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

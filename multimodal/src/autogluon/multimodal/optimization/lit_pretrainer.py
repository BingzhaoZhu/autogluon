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
import numpy as np

class InfoNCELoss(nn.Module):
    r"""InfoNCE Loss. Loss applied during the Contrastive Denoising Self
    Supervised Pre-training routine available in this library
    :information_source: **NOTE**: This loss is in principle not exposed to
     the user, as it is used internally in the library, but it is included
     here for completion.
    See [SAINT: Improved Neural Networks for Tabular Data via Row Attention
    and Contrastive Pre-Training](https://arxiv.org/abs/2106.01342) and
    references therein
    Partially inspired by the code in this [repo](https://github.com/RElbers/info-nce-pytorch)
    Parameters:
    -----------
    temperature: float, default = 0.1
        The logits are divided by the temperature before computing the loss value
    reduction: str, default = "mean"
        Loss reduction method
    """

    def __init__(self, temperature: float = 0.1, reduction: str = "mean"):

        super(InfoNCELoss, self).__init__()

        self.temperature = temperature
        self.reduction = reduction

    def forward(self, z_i, z_j):
        r"""
        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        z, z_ = z_i, z_j

        norm_z = F.normalize(z, dim=-1).flatten(1)
        norm_z_ = F.normalize(z_, dim=-1).flatten(1)

        logits = (norm_z @ norm_z_.t()) / self.temperature
        logits_ = (norm_z_ @ norm_z.t()) / self.temperature

        # the target/labels are the entries on the diagonal
        target = torch.arange(len(norm_z), device=norm_z.device)

        loss = F.cross_entropy(logits, target, reduction=self.reduction)
        loss_ = F.cross_entropy(logits_, target, reduction=self.reduction)

        return (loss + loss_) / 2.0

class NTXent(nn.Module):
    def __init__(self, temperature=0.1):
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
        self.last_batch = None

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

    # def random_perm(self, batch):
    #     lam = self.corruption_rate
    #     batch_size, = batch[self.model.label_key].size()
    #     batch = copy.deepcopy(batch)
    #
    #     num_features = 0
    #     for permodel in self.model.model:
    #         if hasattr(permodel, "categorical_key"):
    #             num_features += len(batch[permodel.categorical_key])
    #         if hasattr(permodel, "numerical_key"):
    #             _, m = batch[permodel.numerical_key].size()
    #             num_features += m
    #
    #     corruption_mask = torch.from_numpy(
    #         np.random.choice(2, (batch_size, num_features), p=[lam, 1 - lam])).to(
    #         batch[self.model.label_key].device
    #     )
    #
    #     random_idx = torch.randperm(batch_size).to(
    #         batch[self.model.label_key].device
    #     )
    #     feature_idx = 0
    #
    #     for permodel in self.model.model:
    #         if hasattr(permodel, "categorical_key"):
    #             categorical_features = []
    #             for categorical_idx in range(len(batch[permodel.categorical_key])):
    #                 categorical_feature = batch[permodel.categorical_key][categorical_idx]
    #                 random_sample = categorical_feature[random_idx].clone()
    #                 mask = corruption_mask[:, feature_idx]
    #                 feature_idx += 1
    #                 random_sample[mask == 0] = categorical_feature[mask == 0]
    #                 categorical_features.append(random_sample)
    #             batch[permodel.categorical_key] = tuple(categorical_features)
    #         if hasattr(permodel, "numerical_key"):
    #             numerical_features = batch[permodel.numerical_key]
    #             random_sample = numerical_features[random_idx].clone()
    #             _, m = numerical_features.size()
    #             mask = corruption_mask[:, feature_idx:feature_idx+m]
    #             feature_idx += m
    #             random_sample[mask == 0] = numerical_features[mask == 0]
    #             batch[permodel.numerical_key] = random_sample
    #     return batch

    def random_perm(self, batch):
        if self.last_batch is None:
            last_batch = batch
        else:
            last_batch = self.last_batch
        self.last_batch = copy.deepcopy(batch)

        corruption_rate = self.corruption_rate
        batch_size, = batch[self.model.label_key].size()
        last_batch_size, = last_batch[self.model.label_key].size()
        batch = copy.deepcopy(batch)

        num_features = 0
        for permodel in self.model.model:
            if hasattr(permodel, "categorical_key"):
                num_features += len(batch[permodel.categorical_key])
            if hasattr(permodel, "numerical_key"):
                _, m = batch[permodel.numerical_key].size()
                num_features += m

        corruption_mask = torch.zeros(batch_size,
                                      num_features,
                                      dtype=torch.bool,
                                      device=batch[self.model.label_key].device
                                      )
        corruption_len = int(num_features * corruption_rate)
        for i in range(batch_size):
            corruption_idx = torch.randperm(num_features)[:corruption_len]
            corruption_mask[i, corruption_idx] = True
        feature_idx = 0

        for permodel in self.model.model:
            if hasattr(permodel, "categorical_key"):
                categorical_features = []
                for categorical_idx in range(len(batch[permodel.categorical_key])):
                    categorical_feature = batch[permodel.categorical_key][categorical_idx]
                    last_categorical_feature = last_batch[permodel.categorical_key][categorical_idx]
                    random_idx = torch.randint(high=last_batch_size, size=(batch_size,))
                    random_sample = last_categorical_feature[random_idx].clone()
                    positive = torch.where(corruption_mask[:, feature_idx], random_sample, categorical_feature)
                    feature_idx += 1
                    categorical_features.append(positive)
                batch[permodel.categorical_key] = tuple(categorical_features)
            if hasattr(permodel, "numerical_key"):
                numerical_features = batch[permodel.numerical_key]
                last_numerical_features = last_batch[permodel.numerical_key]
                _, m = numerical_features.size()
                indices = torch.argsort(torch.rand(*numerical_features.shape), dim=0)
                indices = indices % last_batch_size
                random_sample = last_numerical_features[indices, torch.arange(m).unsqueeze(0)].clone()
                batch[permodel.numerical_key] = torch.where(corruption_mask[:, feature_idx:feature_idx+m],
                                                            random_sample, numerical_features)
                feature_idx += m
        return batch

    # def random_perm(self, batch):
    #     corruption_rate = self.corruption_rate
    #     batch_size, = batch[self.model.label_key].size()
    #     corruption_len = int(batch_size * corruption_rate)
    #     batch = copy.deepcopy(batch)
    #     for permodel in self.model.model:
    #         if hasattr(permodel, "categorical_key"):
    #             categorical_features = []
    #             for categorical_feature in batch[permodel.categorical_key]:
    #                 random_idx = torch.randint(high=batch_size, size=(batch_size,))
    #                 random_sample = categorical_feature[random_idx].clone()
    #                 corruption_idx = torch.randperm(batch_size)[:corruption_len]
    #                 corruption_mask = torch.zeros_like(categorical_feature, dtype=torch.bool)
    #                 corruption_mask[corruption_idx] = True
    #                 positive = torch.where(corruption_mask, random_sample, categorical_feature)
    #                 categorical_features.append(positive)
    #             batch[permodel.categorical_key] = tuple(categorical_features)
    #         if hasattr(permodel, "numerical_key"):
    #             numerical_features = batch[permodel.numerical_key]
    #             _, m = numerical_features.size()
    #             indices = torch.argsort(torch.rand(*numerical_features.shape), dim=0)
    #             random_sample = numerical_features[indices, torch.arange(m).unsqueeze(0)].clone()
    #             corruption_mask = torch.zeros_like(numerical_features, dtype=torch.bool)
    #             for i in range(m):
    #                 corruption_idx = torch.randperm(batch_size)[:corruption_len]
    #                 corruption_mask[corruption_idx, i] = True
    #             batch[permodel.numerical_key] = torch.where(corruption_mask, random_sample, numerical_features)
    #     return batch


import torch.nn.functional as F

class ReconstructionLoss(nn.Module):

    def __init__(self,):
        super().__init__()

    def loss_fn(self, pred_num, pred_cat, target_num, target_cat, mask_num, mask_cat):

        loss = F.mse_loss(pred_num, target_num)

        if pred_cat:
            for i, p in enumerate(pred_cat):
                loss += F.cross_entropy(p, target_cat[:, i].long())
        return loss
import copy
import logging
from typing import List, Optional
import math
import torch
from torch import nn

from ..constants import AUTOMM, FEATURES, LABEL, LOGITS, WEIGHT, NUMERICAL, CATEGORICAL
from .ft_transformer import CLSToken, FT_Transformer
from .fusion import MultimodalFusionTransformer
from .mlp import MLP
from .utils import init_weights

logger = logging.getLogger(AUTOMM)


class HierarchicalTabularTransformer(nn.Module):

    class Node(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            prefix,
            children: list,
            indices: torch.tensor,
            model
        ):
            super().__init__()
            self.children = children
            self.indices = indices
            self.model = model

            self.prefix = prefix

        @property
        def numerical_key(self):
            return f"{self.prefix}_{NUMERICAL}"

        @property
        def categorical_key(self):
            return f"{self.prefix}_{CATEGORICAL}"

        def _refine_batch(self, batch):
            batch = copy.deepcopy(batch)
            if len(self.children) == 0:  # leaf node
                features_cat, features_con = batch[self.categorical_key], batch[self.numerical_key]
                n_cat, n_con = len(features_cat), features_con.shape[1]
                partition_cat = (features_cat[i] for i in self.indices if i<n_cat)
                partition_con = [features_con[:, i-n_cat] for i in self.indices if i >= n_cat]
                partition_con = torch.stack(partition_con, dim=1)
                batch[self.categorical_key] = partition_cat
                batch[self.numerical_key] = partition_con
            else:
                tokens = []
                for node in self.children:
                    emb = node(batch)[FEATURES]
                    tokens.append(emb[:, -1])
                tokens = torch.stack(tokens, dim=1)
                batch[self.categorical_key] = ()
                batch[self.numerical_key] = tokens

            return batch

        def forward(self, batch):
            batch = self._refine_batch(batch)
            return self.model(batch)

    def __init__(
        self,
        prefix: str,
        num_categories: List[int],
        num_numerical_columns: int,
        tokens_per_level: Optional[int] = 12,
        num_sweeps: Optional[int] = 5,
    ):
        super().__init__()
        logger.debug("initializing HierarchicalTabularTransformer")
        self.prefix = prefix
        self.tokens_per_level = tokens_per_level
        n_tokens = len(num_categories) + num_numerical_columns

        indices = [torch.randperm(n_tokens) for _ in range(num_sweeps)]
        indices = torch.stack(indices, dim=0)
        self.root = self.create_node(indices=indices)

    def create_model(self):
        return MultimodalFusionTransformer(
            prefix=self.prefix,
            models=self.model,
            hidden_features=self.hidden_features,
            num_classes=self.num_classes,
        )

    def create_node(self, indices):
        children = []
        if len(indices) > self.tokens_per_level:  # not leaf node
            n_children = math.ceil(len(indices) / self.tokens_per_level)
            for i in range(n_children):
                start, end = (i - 1) * self.tokens_per_level, i * self.tokens_per_level
                if n_children == i+1:
                    indices_per_child = indices[start:]
                else:
                    indices_per_child = indices[start:end]
                children.append(self.create_node(indices_per_child))

        return self.Node(self.prefix,
                         children=children,
                         indices=indices,
                         model=self.create_model()
                         )

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    def forward(
        self,
        batch: dict,
    ):
        return self.root(batch)
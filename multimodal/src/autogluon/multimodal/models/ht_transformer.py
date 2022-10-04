import copy
import logging
from typing import List, Optional
import math
import torch
from torch import nn

from ..constants import AUTOMM, FEATURES, LABEL, LOGITS, WEIGHT, NUMERICAL, CATEGORICAL
from .ft_transformer import CLSToken, FT_Transformer
from .mlp import MLP
from .utils import init_weights

logger = logging.getLogger(AUTOMM)


class InternalTransformer(nn.Module):
    def __init__(
        self,
        prefix: str,
        in_features: int,
        hidden_features: int,
        num_classes: int,
        n_blocks: Optional[int] = 0,
        attention_n_heads: Optional[int] = 8,
        attention_initialization: Optional[str] = "kaiming",
        attention_normalization: Optional[str] = "layer_norm",
        attention_dropout: Optional[str] = 0.2,
        residual_dropout: Optional[str] = 0.0,
        ffn_activation: Optional[str] = "reglu",
        ffn_normalization: Optional[str] = "layer_norm",
        ffn_d_hidden: Optional[str] = 192,
        ffn_dropout: Optional[str] = 0.0,
        prenormalization: Optional[bool] = True,
        first_prenormalization: Optional[bool] = False,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        head_activation: Optional[str] = "relu",
        head_normalization: Optional[str] = "layer_norm",
        adapt_in_features: Optional[str] = None,
        loss_weight: Optional[float] = None,
        row_attention: Optional[bool] = False,
        n_tokens: Optional[int] = None,
    ):

        super().__init__()
        logger.debug("initializing MultimodalFusionTransformer")
        if loss_weight is not None:
            assert loss_weight > 0

        if row_attention:
            n_tokens = n_tokens + 1 if True else n_tokens  # cls_token is always True
        else:
            n_tokens = None

        self.fusion_transformer = FT_Transformer(
            d_token=in_features,
            n_blocks=n_blocks,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            attention_initialization=attention_initialization,
            attention_normalization=attention_normalization,
            ffn_d_hidden=ffn_d_hidden,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation,
            ffn_normalization=ffn_normalization,
            residual_dropout=residual_dropout,
            prenormalization=prenormalization,
            first_prenormalization=first_prenormalization,
            last_layer_query_idx=None,
            n_tokens=n_tokens,
            kv_compression_ratio=kv_compression_ratio,
            kv_compression_sharing=kv_compression_sharing,
            head_activation=head_activation,
            head_normalization=head_normalization,
            d_out=hidden_features,
            projection=False,
            row_attention=row_attention,
        )

        self.head = FT_Transformer.Head(
            d_in=in_features,
            d_out=num_classes,
            bias=True,
            activation=head_activation,
            normalization=head_normalization,
        )

        self.pretrain = False
        self.pretrain_head = FT_Transformer.Head(
            d_in=in_features,
            d_out=in_features,
            bias=True,
            activation=head_activation,
            normalization=head_normalization,
        ) #if self.pretrain else None

        self.cls_token = CLSToken(
            d_token=in_features,
            initialization="uniform",
        )

        self.out_features = in_features

        # init weights
        self.head.apply(init_weights)

        self.prefix = prefix

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"


    def set_pretrain_status(self, is_pretrain=False):
        self.pretrain = is_pretrain


    def forward(
        self,
        features: dict,
    ):
        features = self.cls_token(features)
        features = self.fusion_transformer(features)

        logits = self.pretrain_head(features) if self.pretrain else self.head(features)
        fusion_output = {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }
        return fusion_output


class HierarchicalTabularTransformer(nn.Module):

    class Node(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            prefix,
            children: nn.ModuleList,
            indices: torch.tensor,
            model
        ):
            super().__init__()
            self.children_ = children
            self.indices = indices
            self.model = model
            self.prefix = prefix

        @property
        def numerical_key(self):
            return f"numerical_transformer_{NUMERICAL}"

        @property
        def categorical_key(self):
            return f"categorical_transformer_{CATEGORICAL}"

        def _refine_batch(self, batch):
            if len(self.children_) == 0:  # leaf node
                batch = [batch[:, i] for i in self.indices]
                batch = torch.stack(batch, dim=1)
            else:
                tokens = []
                for node in self.children_:
                    emb = node(batch)[self.prefix][FEATURES]
                    tokens.append(emb[:, -1])
                batch = torch.stack(tokens, dim=1)
            return batch

        def forward(self, batch):
            batch = self._refine_batch(batch)
            return self.model(batch)

    def __init__(
        self,
        prefix: str,
        models: list,
        hidden_features: int,
        num_classes: int,
        num_categories: List[int],
        num_numerical_columns: int,
        tokens_per_level: Optional[int] = 12,
        num_sweeps: Optional[int] = 5,
        n_blocks: Optional[int] = 0,
        attention_n_heads: Optional[int] = 8,
        attention_initialization: Optional[str] = "kaiming",
        attention_normalization: Optional[str] = "layer_norm",
        attention_dropout: Optional[str] = 0.2,
        residual_dropout: Optional[str] = 0.0,
        ffn_activation: Optional[str] = "reglu",
        ffn_normalization: Optional[str] = "layer_norm",
        ffn_d_hidden: Optional[str] = 192,
        ffn_dropout: Optional[str] = 0.0,
        prenormalization: Optional[bool] = True,
        first_prenormalization: Optional[bool] = False,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        head_activation: Optional[str] = "relu",
        head_normalization: Optional[str] = "layer_norm",
        adapt_in_features: Optional[str] = None,
        loss_weight: Optional[float] = None,
        row_attention: Optional[bool] = False,
    ):
        super().__init__()
        logger.debug("initializing HierarchicalTabularTransformer")
        self.prefix = prefix
        self.tokens_per_level = tokens_per_level
        self.n_tokens = len(num_categories) + num_numerical_columns

        self.num_categories = num_categories
        self.num_numerical_columns = num_numerical_columns
        self.num_classes = num_classes
        self.num_sweeps = num_sweeps

        # n_tokens = num_numerical_columns + len(num_categories)
        # if row_attention:
        #     n_tokens = n_tokens + 1 if True else n_tokens  # cls_token is always True
        # else:
        #     n_tokens = None

        self.loss_weight = loss_weight
        self.model = nn.ModuleList(models)

        raw_in_features = [per_model.out_features for per_model in models]

        if adapt_in_features == "min":
            base_in_feat = min(raw_in_features)
        elif adapt_in_features == "max":
            base_in_feat = max(raw_in_features)
        else:
            raise ValueError(f"unknown adapt_in_features: {adapt_in_features}")

        self.adapter = nn.ModuleList([nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features])

        in_features = base_in_feat

        assert len(self.adapter) == len(self.model)
        self.adapter.apply(init_weights)

        self.config = dict(
            prefix=prefix,
            in_features=in_features,
            hidden_features=hidden_features,
            num_classes=num_classes,
            n_blocks=n_blocks,
            attention_n_heads=attention_n_heads,
            ffn_d_hidden=ffn_d_hidden,
            attention_dropout=attention_dropout,
            residual_dropout=residual_dropout,
            ffn_dropout=ffn_dropout,
            attention_normalization=attention_normalization,
            ffn_normalization=ffn_normalization,
            head_normalization=head_normalization,
            ffn_activation=ffn_activation,
            head_activation=head_activation,
            adapt_in_features=adapt_in_features,
            loss_weight=loss_weight,
            row_attention=row_attention,
        )

        indices = [torch.randperm(self.n_tokens) for _ in range(self.num_sweeps)]
        indices = torch.concat(indices, dim=0)
        # indices = [torch.arange(self.n_tokens) for _ in range(self.num_sweeps)]
        # indices = torch.concat(indices, dim=0)
        # setattr(self, "node", self.create_node(indices=indices))
        # self.nodes = {"node1": self.create_node(indices=indices)}
        self.root = self.create_node(indices=indices)

    def create_model(self, is_leaf, indices, children):
        if is_leaf:
            return InternalTransformer(
                n_tokens=len(indices),
                **self.config,
            )
        else:
            return InternalTransformer(
                n_tokens=len(children),
                **self.config,
            )

    def create_node(self, indices):
        children = []
        is_leaf = len(indices) <= self.tokens_per_level
        if not is_leaf:  # not leaf node
            n_children = math.ceil(len(indices) / self.tokens_per_level)
            for i in range(n_children):
                start, end = i * self.tokens_per_level, (i+1) * self.tokens_per_level
                if n_children == i+1:
                    indices_per_child = indices[start:]
                else:
                    indices_per_child = indices[start:end]
                children.append(self.create_node(indices_per_child))

        node = self.Node(self.prefix,
                         children=nn.ModuleList(children),
                         indices=indices,
                         model=self.create_model(is_leaf, indices, children).to('cuda:0')
                         )
        # self.nodes.append(node)
        return node

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    def forward(
        self,
        batch: dict,
    ):
        multimodal_features = []
        output = {}
        for per_model, per_adapter in zip(self.model, self.adapter):
            per_output = per_model(batch)
            multimodal_feature = per_adapter(per_output[per_model.prefix][FEATURES])
            if multimodal_feature.ndim == 2:
                multimodal_feature = torch.unsqueeze(multimodal_feature, dim=1)
            multimodal_features.append(multimodal_feature)

            if self.loss_weight is not None:
                per_output[per_model.prefix].update({WEIGHT: self.loss_weight})
                output.update(per_output)

        multimodal_features = torch.cat(multimodal_features, dim=1)
        return self.root(multimodal_features)

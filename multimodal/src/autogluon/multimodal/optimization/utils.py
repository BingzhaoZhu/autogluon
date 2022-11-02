import functools
import logging
import re
import warnings
import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchmetrics
from omegaconf import DictConfig, OmegaConf
from pytorch_metric_learning import distances, losses, miners
from torch import nn, optim
from torch.nn import functional as F
from transformers import Adafactor
from transformers.trainer_pt_utils import get_parameter_names

from ..constants import (
    ACC,
    ACCURACY,
    AUTOMM,
    AVERAGE_PRECISION,
    BINARY,
    BIT_FIT,
    COLUMN_FEATURES,
    CONTRASTIVE_LOSS,
    COSINE_EMBEDDING_LOSS,
    COSINE_SIMILARITY,
    CROSS_ENTROPY,
    DIRECT_LOSS,
    F1,
    FEATURES,
    IA3,
    IA3_BIAS,
    IA3_NORM,
    LOG_LOSS,
    LORA,
    LORA_BIAS,
    LORA_NORM,
    MAP,
    MULTICLASS,
    NER,
    NORM_FIT,
    OVERALL_ACCURACY,
    PAIR_MARGIN_MINER,
    PEARSONR,
    QUADRATIC_KAPPA,
    R2,
    REGRESSION,
    RMSE,
    ROC_AUC,
    ROOT_MEAN_SQUARED_ERROR,
    SPEARMANR,
)
from ..utils import MeanAveragePrecision
from .losses import SoftTargetCrossEntropy
from .lr_scheduler import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

logger = logging.getLogger(AUTOMM)


def get_loss_func(
    problem_type: str,
    mixup_active: bool,
    loss_func_name: Optional[str] = None,
):
    """
    Choose a suitable Pytorch loss module based on the provided problem type.

    Parameters
    ----------
    problem_type
        Type of problem.
    mixup_active
        The activation determining whether to use mixup.
    loss_func_name
        The name of the function the user wants to use.

    Returns
    -------
    A Pytorch loss module.
    """
    if problem_type in [BINARY, MULTICLASS]:
        if mixup_active:
            loss_func = SoftTargetCrossEntropy()
        else:
            loss_func = nn.CrossEntropyLoss()
    elif problem_type == REGRESSION:
        if loss_func_name is not None:
            if "bcewithlogitsloss" in loss_func_name.lower():
                loss_func = nn.BCEWithLogitsLoss()
            else:
                loss_func = nn.MSELoss()
        else:
            loss_func = nn.MSELoss()
    elif problem_type == NER:
        loss_func = nn.CrossEntropyLoss(ignore_index=0)
    elif problem_type is None:
        return None
    else:
        raise NotImplementedError

    return loss_func


class CustomF1Score(torchmetrics.F1Score):
    """
    Support computing the f1 score of one class, specified by `pos_label`,
    which means the positive label users are interested in.

    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        average: str = "micro",
        mdmc_average: Optional[str] = None,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        pos_label: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.pos_label = pos_label
        if pos_label is not None:
            assert isinstance(pos_label, int)
            average = None

        super().__init__(
            num_classes=num_classes,
            threshold=threshold,
            average=average,
            mdmc_average=mdmc_average,
            ignore_index=ignore_index,
            top_k=top_k,
            multiclass=multiclass,
            **kwargs,
        )

    def compute(self) -> torch.Tensor:
        f1_score = super().compute()
        if self.pos_label is not None:
            f1_score = f1_score[self.pos_label]
        return f1_score


def get_metric(
    metric_name: str,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
):
    """
    Obtain a torchmerics.Metric from its name.
    Define a customized metric function in case that torchmetrics doesn't support some metric.

    Parameters
    ----------
    metric_name
        Name of metric.
    num_classes
        Number of classes.
    pos_label
        The label (0 or 1) of binary classification's positive class, which is used in some metrics, e.g., AUROC.

    Returns
    -------
    torchmetrics.Metric
        A torchmetrics.Metric object.
    custom_metric_func
        A customized metric function.
    """
    metric_name = metric_name.lower()
    if metric_name in [ACC, ACCURACY, OVERALL_ACCURACY]:
        return torchmetrics.Accuracy(), None
    elif metric_name in [RMSE, ROOT_MEAN_SQUARED_ERROR]:
        return torchmetrics.MeanSquaredError(squared=False), None
    elif metric_name == R2:
        return torchmetrics.R2Score(), None
    elif metric_name == QUADRATIC_KAPPA:
        return (
            torchmetrics.CohenKappa(num_classes=num_classes, weights="quadratic"),
            None,
        )
    elif metric_name == ROC_AUC:
        return torchmetrics.AUROC(pos_label=pos_label), None
    elif metric_name == AVERAGE_PRECISION:
        return torchmetrics.AveragePrecision(pos_label=pos_label), None
    elif metric_name in [LOG_LOSS, CROSS_ENTROPY]:
        return torchmetrics.MeanMetric(), functools.partial(F.cross_entropy, reduction="none")
    elif metric_name == COSINE_EMBEDDING_LOSS:
        return torchmetrics.MeanMetric(), functools.partial(F.cosine_embedding_loss, reduction="none")
    elif metric_name == PEARSONR:
        return torchmetrics.PearsonCorrCoef(), None
    elif metric_name == SPEARMANR:
        return torchmetrics.SpearmanCorrCoef(), None
    elif metric_name == F1:
        return CustomF1Score(num_classes=num_classes, pos_label=pos_label), None
    elif metric_name == MAP.lower():
        return (
            MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False),
            None,
        )  # TODO: remove parameter hardcodings here, and add class_metrics
    elif metric_name == DIRECT_LOSS:
        return (
            torchmetrics.MeanMetric(nan_strategy="warn"),
            None,
        )  # This only works for detection where custom_metric is not required for BaseAggregator
    else:
        raise ValueError(f"Unknown metric {metric_name}")


def get_optimizer(
    optim_type: str,
    optimizer_grouped_parameters,
    lr: float,
    weight_decay: float,
    eps: Optional[float] = 1e-6,
    betas: Optional[Tuple[float, float]] = (0.9, 0.999),
    momentum: Optional[float] = 0.9,
):
    """
    Choose a Pytorch optimizer based on its name.

    Parameters
    ----------
    optim_type
        Name of optimizer.
    optimizer_grouped_parameters
        The model parameters to be optimized.
    lr
        Learning rate.
    weight_decay
        Optimizer weight decay.
    eps
        Optimizer eps.
    betas
        Optimizer betas.
    momentum
        Momentum used in the SGD optimizer.

    Returns
    -------
    A Pytorch optimizer.
    """
    if optim_type == "adamw":
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            betas=betas,
        )
    elif optim_type == "adam":
        optimizer = optim.Adam(
            optimizer_grouped_parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optim_type == "sgd":
        optimizer = optim.SGD(
            optimizer_grouped_parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optim_type == "adafactor":
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=lr,
            weight_decay=weight_decay,
            scale_parameter=True,  # Generally recommended to enable scaling
            relative_step=False,
            warmup_init=False,
        )
    else:
        raise ValueError(f"unknown optimizer: {optim_type}")

    return optimizer


def get_lr_scheduler(
    optimizer: optim.Optimizer,
    num_max_steps: int,
    num_warmup_steps: int,
    lr_schedule: str,
    end_lr: Union[float, int],
):
    """
    Get the learning rate scheduler from its name. Here we use our defined learning rate
    scheduler instead of those imported from "transformers" because we want to support
    Pytorch lightning's "ddp_spawn" training strategy.

    Parameters
    ----------
    optimizer
        A Pytorch optimizer.
    num_max_steps
        Number of maximum training steps.
    num_warmup_steps
        Number of steps to do learning rate warmup.
    lr_schedule
        Name of the learning rate scheduler.
    end_lr
        The final learning rate after decay.

    Returns
    -------
    A learning rate scheduler.
    """
    if lr_schedule == "cosine_decay":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_max_steps,
        )
    elif lr_schedule == "polynomial_decay":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_max_steps,
            lr_end=end_lr,
            power=1,
        )
    elif lr_schedule == "linear_decay":
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_max_steps,
        )
    else:
        raise ValueError(f"unknown lr schedule: {lr_schedule}")

    return scheduler


def get_weight_decay_param_names(model: nn.Module):
    """
    Set the layer normalization parameters and other layers' bias parameters not to use weight decay.

    Parameters
    ----------
    model
        A Pytorch model.

    Returns
    -------
    A list of parameter names not using weight decay.
    """
    # By default, we should not apply weight decay for all the norm layers
    decay_param_names = get_parameter_names(
        model,
        [nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm],
    )
    decay_param_names = [
        name
        for name in decay_param_names
        if (
            "bias" not in name
            and "cls_token" not in name
            and "categorical_feature_tokenizer" not in name
            and "numerical_feature_tokenizer" not in name
        )
    ]
    return decay_param_names


def get_norm_layer_param_names(model: nn.Module):
    """
    Get parameters associated with the normalization layers

    Parameters
    ----------
    model
        A Pytorch model

    Returns
    -------
    norm_param_names
        A list of normalization parameter names
    """
    all_param_names = [name for name, _ in model.named_parameters()]
    all_param_names_except_norm_names = get_parameter_names(
        model,
        [nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm],
    )
    norm_param_names = [name for name in all_param_names if name not in all_param_names_except_norm_names]
    return norm_param_names


def apply_single_lr(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    return_params: Optional[bool] = True,
):
    """
    Set to use a single learning rate for all parameters. Layer normalization parameters and other
    layers' bias parameters don't use weight decay.

    Parameters
    ----------
    model
        A Pytorch model.
    lr
        Learning rate.
    weight_decay
        Weight decay.
    return_params
        Whether to return parameters or their names. If you want to double-check
        whether the learning rate setup is as expected, you can set "return_params=False",
        and print the layer names along with their learning rates through
        "print("Param groups = %s" % json.dumps(optimizer_grouped_parameters, indent=2))".

    Returns
    -------
    The grouped parameters or their names.
    """
    decay_param_names = get_weight_decay_param_names(model)
    optimizer_grouped_parameters = [
        {
            "params": [p if return_params else n for n, p in model.named_parameters() if n in decay_param_names],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [p if return_params else n for n, p in model.named_parameters() if n not in decay_param_names],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
    return optimizer_grouped_parameters


def apply_two_stages_lr(
    model: nn.Module,
    lr: float,
    lr_mult: Union[float, int],
    weight_decay: float,
    return_params: Optional[bool] = True,
):
    """
    Set up the pretrained backbone to use a smaller learning rate (lr * lr_mult).
    The newly added head layers use the normal learning rate (lr).
    Layer normalization parameters and other layers' bias parameters don't use weight decay.

    Parameters
    ----------
    model
        A Pytorch model.
    lr
        The learning rate.
    lr_mult
        The multiplier (0, 1) to scale down the learning rate.
    weight_decay
        Weight decay.
    return_params
        return_params
        Whether to return parameters or their names. If you want to double-check
        whether the learning rate setup is as expected, you can set "return_params=False",
        and print the layer names along with their learning rates through
        "print("Param groups = %s" % json.dumps(optimizer_grouped_parameters, indent=2))".

    Returns
    -------
    The grouped parameters or their names.
    """
    decay_param_names = get_weight_decay_param_names(model)

    optimizer_grouped_parameters = [
        {
            "params": [
                p if return_params else n
                for n, p in model.named_parameters()
                if n in decay_param_names and not any(bb in n for bb in model.head_layer_names)
            ],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [
                p if return_params else n
                for n, p in model.named_parameters()
                if n not in decay_param_names and not any(bb in n for bb in model.head_layer_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p if return_params else n
                for n, p in model.named_parameters()
                if n in decay_param_names and any(bb in n for bb in model.head_layer_names)
            ],
            "weight_decay": weight_decay,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p if return_params else n
                for n, p in model.named_parameters()
                if n not in decay_param_names and any(bb in n for bb in model.head_layer_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    return optimizer_grouped_parameters


def get_trainable_params_efficient_finetune(
    norm_param_names: List[str],
    efficient_finetune: Optional[str] = None,
):
    """
     Get the list of trainable parameters according to the provided efficient finetuning method.

    Parameters
    ----------
    norm_param_names
        The parameters associated with the normalization layers
    efficient_finetune
        Efficient finetuning strategy. Trainable parameters will be adjusted according to the method.
    trainable_param_names
        Initial specification of layers that should be trained.

    Returns
    -------
    Get list of trainable parameter names according to the provided efficient finetuning method.
    """
    trainable_param_names = []

    if efficient_finetune == BIT_FIT:
        trainable_param_names.append(".*bias*.")
    elif efficient_finetune == NORM_FIT:
        trainable_param_names.append(".*bias*.")
        trainable_param_names += norm_param_names
    elif efficient_finetune in [LORA, IA3]:
        trainable_param_names.append(".*lora_*.")
    elif efficient_finetune in [LORA_BIAS, IA3_BIAS]:
        trainable_param_names.append(".*lora_*.")
        trainable_param_names.append(".*bias*.")
    elif efficient_finetune in [LORA_NORM, IA3_NORM]:
        trainable_param_names.append(".*lora_*.")
        trainable_param_names.append(".*bias*.")
        trainable_param_names += norm_param_names
    elif efficient_finetune is not None and efficient_finetune != "None":
        raise NotImplementedError(
            f"The efficient finetuning strategy '{efficient_finetune}'"
            f" is not supported. We only support"
            f" '{BIT_FIT}', '{NORM_FIT}', '{LORA}', '{LORA_NORM}', '{LORA_BIAS}', '{IA3}', '{IA3_BIAS}', '{IA3_NORM}'."
        )

    return trainable_param_names


def apply_layerwise_lr_decay(
    model: nn.Module,
    lr: float,
    lr_decay: float,
    weight_decay: float,
    efficient_finetune: Optional[str] = None,
    trainable_param_names: Optional[List] = None,
):
    """
    Assign monotonically decreasing learning rates for layers from the output end to the input end.
    The intuition behind is that later layers are more task-related compared to the early layers.
    Layer normalization parameters and other layers' bias parameters don't use weight decay.
    If you want to double-check whether the learning rate setup is as expected,
    you can print the layer names along with their learning rates through
    "print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))".

    Parameters
    ----------
    model
        A Pytorch model.
    lr
        The learning rate.
    lr_decay
        The learning rate decay factor (0, 1).
    weight_decay
        Weight decay.
    efficient_finetune
        Efficient finetuning strategy. It will only finetune part of the parameters

    Returns
    -------
    The grouped parameters based on their layer ids and whether using weight decay.
    """
    parameter_group_names = {}
    parameter_group_vars = {}
    decay_param_names = get_weight_decay_param_names(model)

    for name, param in model.named_parameters():
        layer_id = model.name_to_id[name]
        if layer_id == 0:  # Set top layer (e.g. head, fusion_mlp, adapter) as being trainable.
            param.requires_grad = True
        elif (
            efficient_finetune is not None
            and efficient_finetune != "None"
            and trainable_param_names
            and not any([re.match(trainable_param_name, name) for trainable_param_name in trainable_param_names])
        ):
            param.requires_grad = False

        if not param.requires_grad:
            continue  # frozen weights

        if name in decay_param_names:
            group_name = "decay"
            this_weight_decay = weight_decay
        else:
            group_name = "no_decay"
            this_weight_decay = 0.0

        layer_id = model.name_to_id[name]
        group_name = "layer_%d_%s" % (layer_id, group_name)

        if group_name not in parameter_group_names:
            scale = lr_decay**layer_id
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": scale * lr,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": scale * lr,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values())


def gather_column_features(
    output: Dict[str, Dict],
    column_names: Union[str, List[str]],
):
    """
    Gather column features from models' outputs.
    For each feature name in one model's output, we enumerate the provided column names to see
    whether (partial) the provided columns share one cls feature or they have independent features.

    TODO: return features' masks and use them to filter the losses.

    Parameters
    ----------
    output
        The models' outputs.
    column_names
        The columns whose features we want to get.

    Returns
    -------
    The gathered feature vectors. Each sample should only have one feature vector.
    """
    if isinstance(column_names, str):
        column_names = [column_names]

    gathered_features = []
    # logger.debug(f"gather features for columns: {column_names}")
    for per_model_name, per_model_output in output.items():
        # logger.debug(f"gather column features from model: {per_model_name}")
        for feature_name in per_model_output[COLUMN_FEATURES][FEATURES]:
            # logger.debug(f"processing feature: {feature_name}")
            columns_share_one_feature = []
            for col_name in column_names:
                if col_name in feature_name:
                    # this column feature is part of the cls feature
                    if not (feature_name.startswith(col_name) and feature_name.endswith(col_name)):
                        columns_share_one_feature.append(col_name)
                        # logger.debug(f"column {col_name} is included in feature {feature_name}")
                    else:  # this column's feature is independent of other columns'
                        gathered_features.append(per_model_output[COLUMN_FEATURES][FEATURES][col_name])
                        # logger.debug(f"col_name {col_name} has an independent feature in model: {per_model_name}")

            # two or more columns share one cls feature, and no other columns share it.
            if len(columns_share_one_feature) > 0:
                assert len("_".join(columns_share_one_feature)) == len(
                    feature_name
                ), f"model `{per_model_name}`'s cls feature name `{feature_name}` doesn't match `{columns_share_one_feature}`"
                gathered_features.append(per_model_output[COLUMN_FEATURES][FEATURES][feature_name])

    if len(gathered_features) > 1:
        # currently only support features of the same shape
        assert all(
            per_features.shape == gathered_features[0].shape for per_features in gathered_features
        ), "Currently we only support gathering features of the same dimension."

    if len(gathered_features) == 0:
        raise ValueError(f"No features are found for columns names {column_names}.")

    gathered_features = torch.stack(gathered_features, dim=0).mean(dim=0)  # (b, d)

    return gathered_features


def get_metric_learning_distance_func(
    name: str,
):
    """
    Return one pytorch metric learning's distance function based on its name.

    Parameters
    ----------
    name
        distance function name

    Returns
    -------
    A distance function from the pytorch metric learning package.
    """
    if name.lower() == COSINE_SIMILARITY:
        return distances.CosineSimilarity()
    else:
        raise ValueError(f"Unknown distance measure: {name}")


def infer_matcher_loss(data_format: str, problem_type: str):
    """
    Infer the loss type to train the matcher.

    Parameters
    ----------
    data_format
        The training data format, e.g., pair or triplet.
    problem_type
        Type of problem.

    Returns
    -------
    The loss name.
    """
    if data_format == "pair":
        if problem_type is None:
            return ["multi_negatives_softmax_loss"]
        elif problem_type == BINARY:
            return ["contrastive_loss"]
        elif problem_type == REGRESSION:
            return ["cosine_similarity_loss"]
        else:
            raise ValueError(f"Unsupported data format {data_format} with problem type {problem_type}")
    elif data_format == "triplet":
        if problem_type is None:
            return ["multi_negatives_softmax_loss"]
        else:
            raise ValueError(f"Unsupported data format {data_format} with problem type {problem_type}")
    else:
        raise ValueError(f"Unsupported data format: {data_format}")


def get_matcher_loss_func(
    data_format: str,
    problem_type: str,
    loss_type: Optional[str] = None,
    pos_margin: Optional[float] = None,
    neg_margin: Optional[float] = None,
    distance_type: Optional[str] = None,
):
    """
    Return a list of pytorch metric learning's loss functions based on their names.

    Parameters
    ----------
    data_format
        The training data format, e.g., pair or triplet.
    problem_type
        Type of problem.
    loss_type
        The provided loss type.
    pos_margin
        The positive margin in computing the metric learning loss.
    neg_margin
        The negative margin in computing the metric learning loss.
    distance_type
        The distance function type.

    Returns
    -------
    A loss function of metric learning.
    """

    allowable_loss_types = infer_matcher_loss(data_format=data_format, problem_type=problem_type)
    if loss_type is not None:
        assert loss_type in allowable_loss_types, f"data format {data_format} can't use loss {loss_type}."
    else:
        loss_type = allowable_loss_types[0]

    if loss_type.lower() == CONTRASTIVE_LOSS:
        return losses.ContrastiveLoss(
            pos_margin=pos_margin,
            neg_margin=neg_margin,
            distance=get_metric_learning_distance_func(distance_type),
        )
    else:
        raise ValueError(f"Unknown metric learning loss: {loss_type}")


def get_matcher_miner_func(
    miner_type: str,
    pos_margin: float,
    neg_margin: float,
    distance_type: str,
):
    """
    Return a pytorch metric learning's miner functions based on their names.
    The miners are used to mine the positive and negative examples.

    Parameters
    ----------
    miner_type
        The miner function type.
    pos_margin
        The positive margin used by the miner function.
    neg_margin
        The negative margin used by the miner function.
    distance_type
        The distance function type.

    Returns
    -------
    A miner function to mine positive and negative samples.
    """
    if miner_type.lower() == PAIR_MARGIN_MINER:
        return miners.PairMarginMiner(
            pos_margin=pos_margin,
            neg_margin=neg_margin,
            distance=get_metric_learning_distance_func(distance_type),
        )
    else:
        raise ValueError(f"Unknown metric learning miner: {miner_type}")


def generate_metric_learning_labels(
    num_samples: int,
    match_label: int,
    labels: torch.Tensor,
):
    """
    Generate labels to compute the metric learning loss of one mini-batch.
    For n samples, it generates 2*n labels since each match has two sides, each of which
    has one label. If we know the matching label, then it determines the two sides' labels
    according to whether their label is the matching label. If the matching label is None,
    it assigns a unique label for each side.

    Parameters
    ----------
    num_samples
        number of samples.
    match_label
        The matching label, which can be None.
    labels
        The sample labels used in the supervised setting. It's required only when match_label is not None.

    Returns
    -------
    The labels used in computing the metric learning loss.
    """
    labels_1 = torch.arange(num_samples)

    if match_label is not None:
        labels_2 = torch.arange(num_samples, num_samples * 2)
        # users need to specify the match_label based on the raw label's semantic meaning.
        mask = labels == match_label
        labels_2[mask] = labels_1[mask]
    else:
        labels_2 = torch.arange(num_samples)

    metric_learning_labels = torch.cat([labels_1, labels_2], dim=0)

    return metric_learning_labels


def compute_probability(
    logits: Optional[torch.Tensor] = None,
    embeddings1: Optional[torch.Tensor] = None,
    embeddings2: Optional[torch.Tensor] = None,
    reverse_prob: Optional[bool] = False,
):
    """
    Compute probabilities from logits or embedding pairs.

    Parameters
    ----------
    logits
        The output of a model's head layer.
    embeddings1
        Feature embeddings of one side in matching.
    embeddings2
        Feature embeddings 2 of the other side in matching.
    reverse_prob
        Whether to reverse the probability.

    Returns
    -------
    Probabilities.
    """
    if logits is not None:
        prob = F.softmax(logits.float(), dim=1)[:, 1]
    else:
        cosine_similarity = F.cosine_similarity(embeddings1, embeddings2)
        prob = 0.5 * (cosine_similarity + 1)

    if reverse_prob:
        prob = 1 - prob

    return prob


class ReconstructionLoss:
    def __init__(self, model):
        self.model = model

    def __call__(self, batch, batch_):
        batch_ = batch_[self.model.prefix]['logits']
        loss = 0
        for permodel in self.model.model:
            if hasattr(permodel, "categorical_key"):
                for y_, y in zip(batch_["cat_out"], batch[permodel.categorical_key]):
                    loss += F.cross_entropy(y_, y.long())
            if hasattr(permodel, "numerical_key"):
                y = batch[permodel.numerical_key]
                y_ = batch_["num_out"]
                loss += F.mse_loss(y_, y)
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
        elif self.mode == "permutation":
            return self.random_perm(batch)
        else:
            raise ValueError(
                f"Current mode {self.mode} is not supported."
                "Consider choosing from the following options:"
                "identical, random_perm."
            )

    def identical(self, batch):
        batch = copy.deepcopy(batch)
        return batch

    def random_perm(self, batch):
        corruption_rate = self.corruption_rate
        batch_size, = batch[self.model.label_key].size()
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
                for categorical_feature in batch[permodel.categorical_key]:
                    random_idx = torch.randint(high=batch_size, size=(batch_size,))
                    random_sample = categorical_feature[random_idx].clone()
                    positive = torch.where(corruption_mask[:, feature_idx], random_sample, categorical_feature)
                    feature_idx += 1
                    categorical_features.append(positive)
                batch[permodel.categorical_key] = tuple(categorical_features)
            if hasattr(permodel, "numerical_key"):
                numerical_features = batch[permodel.numerical_key]
                _, m = numerical_features.size()
                indices = torch.randint(high=batch_size, size=(batch_size, m))
                random_sample = numerical_features[indices, torch.arange(m).unsqueeze(0)].clone()
                batch[permodel.numerical_key] = torch.where(corruption_mask[:, feature_idx:feature_idx+m],
                                                            random_sample, numerical_features)
                feature_idx += m
        return batch


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
        z_i, z_j = z_i.flatten(1), z_j.flatten(1)

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        # mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float().to(z_i.get_device())
        numerator = torch.exp(positives / self.temperature)

        denominator = torch.exp(similarity / self.temperature) # * mask
        all_losses = -torch.log(numerator / torch.mean(denominator, dim=1))

        # all_losses = -torch.log(numerator)
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss


class NTXent_distill(nn.Module):
    def __init__(self, temperature=1):
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
        z_i, z_j = z_i.flatten(1), z_j.flatten(1)

        if z_i.size(1) == 1:
            return F.mse_loss(z_i, z_j)
        else:
            z_i, z_j = z_i / self.temperature, z_j / self.temperature
            z_i = F.softmax(z_i, dim=-1)
            return F.cross_entropy(z_j, z_i)
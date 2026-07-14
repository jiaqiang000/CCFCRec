import math
import os
import sys
import pickle
import random
import functools
import json
import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from preprocess import serial_asin_category
from extract_img_feature import get_img_feature_pickle
from support import RatingDataset
from tqdm import tqdm
import pandas as pd
import time
import numpy as np
from support import serialize_user
from test import Validate
from myargs import get_args, args_tostring
from m11_features import (
    M11_FEATURE_MODE_FULL_STRUCTURAL,
    M11_FEATURE_MODE_TARGET_MASKED,
    M11_FEATURE_WIDTH,
    load_m11_feature_tensor,
)
from cicp_features import CICP_FEATURE_WIDTH, load_cicp_feature_tensor


def resolve_device():
    requested = os.environ.get('CCFCREC_DEVICE', '').strip().lower()
    if requested == 'cpu':
        return torch.device('cpu')
    if requested == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if requested == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    if requested:
        raise RuntimeError(f"Unsupported or unavailable CCFCREC_DEVICE={requested}")
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


device = resolve_device()
if device.type == 'cuda':
    torch.cuda.set_device(int(os.environ.get('CCFCREC_CUDA_DEVICE', '0')))


TASK4_WEIGHT_METHOD_VARIANTS = {
    "task4_rsp_high_weight",
    "task4_acat_high_weight",
    "task4_acat_shuffle_high_weight",
    "task4_acat_trainhard_weight",
    "task4_highdetail_trainhard_weight",
    "task4_highdetail_trainhard_shuffle_weight",
    "m11r2_qbpr_score_weight",
    "m11r2_qbpr_curriculum",
}

M11R2_QBPR_WEIGHT_METHOD_VARIANTS = {
    "m11r2_qbpr_score_weight",
    "m11r2_qbpr_curriculum",
}

M11R2_FOCAL_METHOD_VARIANTS = {"m11r2_qbpr_focal"}
M11R3_DUAL_RESIDUAL_METHOD_VARIANTS = {"m11r3_dual_residual"}
M11R3_NORM_CAPPED_METHOD_VARIANTS = {"m11r3_norm_capped_residual"}
M11R3_NEIGHBOR_TRANSFER_METHOD_VARIANTS = {"m11r3_neighbor_transfer"}
M11R3_FILM_METHOD_VARIANTS = {"m11r3_target_film"}
M11R4_PROTECTED_EXPERT_METHOD_VARIANTS = {"m11r4_protected_experts"}
M11R4_CONTINUOUS_FUSION_METHOD_VARIANTS = {"m11r4_continuous_fusion"}
M11R4_RELATIONAL_ALIGNMENT_METHOD_VARIANTS = {"m11r4_relational_alignment"}
M11R4_CONTINUOUS_FOCAL_METHOD_VARIANTS = {"m11r4_continuous_focal"}
CICPR1_E4_RESIDUAL_METHOD_VARIANTS = {"cicpr1_e4_residual"}
CICPR1_MODALITY_ROUTING_METHOD_VARIANTS = {"cicpr1_modality_routing"}
CICPR1_CATEGORY_EXPERT_METHOD_VARIANTS = {"cicpr1_category_expert"}
CICPR1_ALIGNMENT_METHOD_VARIANTS = {"cicpr1_alignment_curriculum"}
CICPR1_COUNTERFACTUAL_METHOD_VARIANTS = {"cicpr1_counterfactual_margin"}
CICPR1_ADAPTIVE_ATTENTION_METHOD_VARIANTS = {"cicpr1_adaptive_attention"}
CICPR1_METHOD_VARIANTS = (
    CICPR1_E4_RESIDUAL_METHOD_VARIANTS
    | CICPR1_MODALITY_ROUTING_METHOD_VARIANTS
    | CICPR1_CATEGORY_EXPERT_METHOD_VARIANTS
    | CICPR1_ALIGNMENT_METHOD_VARIANTS
    | CICPR1_COUNTERFACTUAL_METHOD_VARIANTS
    | CICPR1_ADAPTIVE_ATTENTION_METHOD_VARIANTS
)
CICPR2_CONTENT_DIRECTION_RESIDUAL_METHOD_VARIANTS = {
    "cicpr2_content_direction_residual"
}
CICPR2_CATEGORY_INCREMENT_METHOD_VARIANTS = {"cicpr2_category_increment_gate"}
CICPR2_CROSS_MODAL_ATTENTION_METHOD_VARIANTS = {"cicpr2_cross_modal_attention"}
CICPR2_SCORE_DISTILLATION_METHOD_VARIANTS = {"cicpr2_score_distillation"}
CICPR2_ORDINAL_COUNTERFACTUAL_METHOD_VARIANTS = {
    "cicpr2_ordinal_counterfactual"
}
CICPR2_RELIABILITY_DROPOUT_METHOD_VARIANTS = {"cicpr2_reliability_dropout"}
CICPR2_METHOD_VARIANTS = (
    CICPR2_CONTENT_DIRECTION_RESIDUAL_METHOD_VARIANTS
    | CICPR2_CATEGORY_INCREMENT_METHOD_VARIANTS
    | CICPR2_CROSS_MODAL_ATTENTION_METHOD_VARIANTS
    | CICPR2_SCORE_DISTILLATION_METHOD_VARIANTS
    | CICPR2_ORDINAL_COUNTERFACTUAL_METHOD_VARIANTS
    | CICPR2_RELIABILITY_DROPOUT_METHOD_VARIANTS
)
CICP_METHOD_VARIANTS = CICPR1_METHOD_VARIANTS | CICPR2_METHOD_VARIANTS
M11R4_FEATURE_METHOD_VARIANTS = (
    M11R4_PROTECTED_EXPERT_METHOD_VARIANTS
    | M11R4_CONTINUOUS_FUSION_METHOD_VARIANTS
    | M11R4_RELATIONAL_ALIGNMENT_METHOD_VARIANTS
    | M11R4_CONTINUOUS_FOCAL_METHOD_VARIANTS
)
M11R3_FEATURE_METHOD_VARIANTS = (
    M11R3_DUAL_RESIDUAL_METHOD_VARIANTS
    | M11R3_NORM_CAPPED_METHOD_VARIANTS
    | M11R3_NEIGHBOR_TRANSFER_METHOD_VARIANTS
    | M11R3_FILM_METHOD_VARIANTS
)
M11R2_FEATURE_METHOD_VARIANTS = (
    {"m11r2_target_feature_fusion"}
    | M11R3_FEATURE_METHOD_VARIANTS
    | M11R4_FEATURE_METHOD_VARIANTS
)
M11R3_UNMASKED_FEATURE_METHOD_VARIANTS = (
    M11R3_DUAL_RESIDUAL_METHOD_VARIANTS
    | M11R3_NEIGHBOR_TRANSFER_METHOD_VARIANTS
    | M11R3_FILM_METHOD_VARIANTS
    | M11R4_FEATURE_METHOD_VARIANTS
)

TASK4_PAIR_MARGIN_METHOD_VARIANTS = {
    "task4_acat_pairmargin_weight",
    "task4_acat_rsp_residual_pairmargin",
    "task4_acat_hardonly_qmargin",
    "task4_highdetail_pairmargin",
    "task4_highdetail_pairmargin_shuffle",
}

TASK4_COMPETITOR_PAIR_METHOD_VARIANTS = {
    "task4_competitor_pair",
    "task4_competitor_pair_shuffle",
    "task4_competitor_pair_rsp_control",
    "task4_competitor_pair_acat_control",
    "task4_boundary_competitor_pair",
    "task4_boundary_competitor_pair_shuffle",
    "task4_boundary_competitor_pair_rsp_control",
    "task4_boundary_competitor_pair_acat_control",
    "m11_target_competitor_pair",
    "m11_target_competitor_pair_shuffle",
    "m11_target_competitor_pair_lowrsp_control",
    "m11_target_competitor_pair_rsp_control",
    "m11r1_full_target_competitor_pair",
    "m11r1_popmatch_competitor_pair_control",
    "m11r1_lowacat_competitor_pair_control",
}

TASK4_BOUNDARY_COMPETITOR_PAIR_METHOD_VARIANTS = {
    "task4_boundary_competitor_pair",
    "task4_boundary_competitor_pair_shuffle",
    "task4_boundary_competitor_pair_rsp_control",
    "task4_boundary_competitor_pair_acat_control",
}

TASK4_METHOD_VARIANTS = (
    TASK4_WEIGHT_METHOD_VARIANTS
    | TASK4_PAIR_MARGIN_METHOD_VARIANTS
    | TASK4_COMPETITOR_PAIR_METHOD_VARIANTS
    | M11R2_FOCAL_METHOD_VARIANTS
    | M11R2_FEATURE_METHOD_VARIANTS
)

TASK4_FORBIDDEN_TRAIN_COLUMNS = {
    "hr@20",
    "ndcg@20",
    "baseline_ndcg@20",
    "margin_proxy",
    "baseline_margin_proxy",
    "best_target_rank",
    "baseline_best_target_rank",
    "eval_baseline_hard_flag",
    "high_acat_eval_hard_flag",
    "proxy_ensemble_score",
    "proxy_ensemble_score_x",
    "proxy_ensemble_score_y",
    "consensus_score",
    "consensus_score_x",
    "consensus_score_y",
    "delta_ndcg@20",
}


def set_random_seed(seed):
    if seed is None or seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id, base_seed):
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_category_reweight(item_genres, args):
    if args.method_variant != 'weak_q_reweight':
        return None
    if args.weak_loss_alpha <= 0:
        return None
    category_count = (item_genres != -1).sum(dim=1).float()
    weights = torch.ones_like(category_count)
    weights = torch.where(category_count <= args.weak_cat_threshold, weights + args.weak_loss_alpha, weights)
    return weights / weights.mean().detach()


def build_adaptive_qbpr_weights(item_genres, support_confidence, args):
    if getattr(args, "method_variant", "baseline") != "adaptive_conf_qbpr":
        return None
    if getattr(args, "adaptive_loss_alpha", 0) <= 0:
        return None
    category_count = (item_genres != -1).sum(dim=1).float()
    weak_mask = category_count <= args.weak_cat_threshold
    support_confidence = torch.clamp(support_confidence.float().to(category_count.device), min=0.0, max=1.0)
    weights = torch.ones_like(category_count)
    weights = weights + args.adaptive_loss_alpha * weak_mask.float() * support_confidence
    return weights / weights.mean().detach()


def uses_task4_item_weights(args):
    return getattr(args, "method_variant", "baseline") in TASK4_WEIGHT_METHOD_VARIANTS


def uses_task4_pair_margin(args):
    return getattr(args, "method_variant", "baseline") in TASK4_PAIR_MARGIN_METHOD_VARIANTS


def uses_task4_competitor_pair(args):
    return getattr(args, "method_variant", "baseline") in TASK4_COMPETITOR_PAIR_METHOD_VARIANTS


def uses_task4_boundary_competitor_pair(args):
    return getattr(args, "method_variant", "baseline") in TASK4_BOUNDARY_COMPETITOR_PAIR_METHOD_VARIANTS


def uses_m11r2_focal_qbpr(args):
    return getattr(args, "method_variant", "baseline") in M11R2_FOCAL_METHOD_VARIANTS


def uses_m11r2_feature_fusion(args):
    return getattr(args, "method_variant", "baseline") in M11R2_FEATURE_METHOD_VARIANTS


def uses_cicp_features(args):
    return getattr(args, "method_variant", "baseline") in CICP_METHOD_VARIANTS


def resolve_m11_feature_mode(args):
    method_variant = getattr(args, "method_variant", "baseline")
    if method_variant in M11R3_UNMASKED_FEATURE_METHOD_VARIANTS:
        return M11_FEATURE_MODE_FULL_STRUCTURAL
    return M11_FEATURE_MODE_TARGET_MASKED


def _bool_series(series):
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y", "t"})


def _task4_high_acat_flags(profile):
    if "high_acat_flag" in profile.columns:
        return _bool_series(profile["high_acat_flag"])
    if "s_cat_v3_group" in profile.columns:
        return profile["s_cat_v3_group"].astype(str).eq("s_cat_v3_strong")
    raise ValueError("Task4 profile 缺少 high_acat_flag 或 s_cat_v3_group")


def _task4_high_detail_flags(profile):
    if "cat_count_bin" in profile.columns:
        return profile["cat_count_bin"].astype(str).eq("cat_count_5_plus")
    if "category_count" in profile.columns:
        return _numeric_task4_series(profile, "category_count").ge(5)
    raise ValueError("Task4 profile 缺少 cat_count_bin 或 category_count")


def _task4_highdetail_trainhard_flags(profile):
    if "high_acat_train_safe_hard_flag" not in profile.columns:
        raise ValueError("Task4 profile 缺少 high_acat_train_safe_hard_flag")
    return _task4_high_detail_flags(profile) & _bool_series(profile["high_acat_train_safe_hard_flag"])


def _task4_shuffle_by_split_and_detail(profile, flags, args, scores=None):
    shuffle_seed = int(getattr(args, "task4_shuffle_seed", 43))
    rng = np.random.default_rng(shuffle_seed)
    flags = pd.Series(flags, index=profile.index).astype(bool)
    shuffled_flags = pd.Series(False, index=profile.index)
    shuffled_scores = None if scores is None else pd.Series(0.0, index=profile.index, dtype=float)
    group_cols = [col for col in ["split", "cat_count_bin"] if col in profile.columns]
    if not group_cols:
        group_cols = [None]
    groups = profile.groupby(group_cols, dropna=False).groups if group_cols != [None] else {None: profile.index}
    for _, group_index in groups.items():
        group_index = pd.Index(group_index)
        permutation = rng.permutation(len(group_index))
        shuffled_flags.loc[group_index] = flags.loc[group_index].to_numpy(dtype=bool)[permutation]
        if shuffled_scores is not None:
            shuffled_scores.loc[group_index] = pd.Series(scores, index=profile.index).loc[group_index].to_numpy(dtype=float)[permutation]
    if shuffled_scores is None:
        return shuffled_flags
    return shuffled_flags, shuffled_scores


M11R2_QBPR_PROFILE_COLUMNS = {
    "m11r2_qbpr_score_weight": (
        "m11r1_full_target_flag",
        "m11r1_full_target_loss_score",
    ),
    "m11r2_qbpr_curriculum": (
        "m11r1_full_target_flag",
        "m11r1_full_target_loss_score",
    ),
}


def _m11r2_qbpr_flags_and_scores(profile, method_variant):
    if method_variant not in M11R2_QBPR_PROFILE_COLUMNS:
        raise ValueError(f"unsupported M11-R2 qBPR method_variant={method_variant}")
    flag_column, score_column = M11R2_QBPR_PROFILE_COLUMNS[method_variant]
    missing = [column for column in [flag_column, score_column] if column not in profile.columns]
    if missing:
        raise ValueError(f"M11-R2 profile 缺少 {missing}")
    flags = _bool_series(profile[flag_column])
    scores = _clip01(_numeric_task4_series(profile, score_column))
    return flags, scores


def _task4_variant_flags(profile, args):
    method_variant = getattr(args, "method_variant", "baseline")
    if method_variant in M11R2_QBPR_WEIGHT_METHOD_VARIANTS:
        flags, _ = _m11r2_qbpr_flags_and_scores(profile, method_variant)
        return flags
    if method_variant == "task4_rsp_high_weight":
        if "RSP_group" not in profile.columns:
            raise ValueError("Task4 profile 缺少 RSP_group")
        return profile["RSP_group"].astype(str).eq("RSP_high")
    if method_variant == "task4_acat_high_weight":
        return _task4_high_acat_flags(profile)
    if method_variant == "task4_acat_trainhard_weight":
        if "high_acat_train_safe_hard_flag" not in profile.columns:
            raise ValueError("Task4 profile 缺少 high_acat_train_safe_hard_flag")
        return _bool_series(profile["high_acat_train_safe_hard_flag"])
    if method_variant == "task4_highdetail_trainhard_weight":
        return _task4_highdetail_trainhard_flags(profile)
    if method_variant == "task4_highdetail_trainhard_shuffle_weight":
        flags = _task4_highdetail_trainhard_flags(profile)
        return _task4_high_detail_flags(profile) & _task4_shuffle_by_split_and_detail(profile, flags, args)
    if method_variant == "task4_acat_shuffle_high_weight":
        shuffle_seed = int(getattr(args, "task4_shuffle_seed", 43))
        rng = np.random.default_rng(shuffle_seed)
        flags = _task4_high_acat_flags(profile).astype(bool)
        shuffled = pd.Series(False, index=profile.index)
        if "split" in profile.columns:
            for _, split_index in profile.groupby("split", dropna=False).groups.items():
                values = flags.loc[split_index].to_numpy(dtype=bool).copy()
                rng.shuffle(values)
                shuffled.loc[split_index] = values
        else:
            values = flags.to_numpy(dtype=bool).copy()
            rng.shuffle(values)
            shuffled.loc[:] = values
        return shuffled
    raise ValueError(f"unsupported Task4 method_variant={method_variant}")


def _numeric_task4_series(profile, column):
    if column not in profile.columns:
        raise ValueError(f"Task4 profile 缺少 {column}")
    return pd.to_numeric(profile[column], errors="coerce").fillna(0.0)


def _clip01(series):
    return _numeric_task4_series(pd.DataFrame({"value": series}), "value").clip(lower=0.0, upper=1.0)


def _task4_pair_margin_flags_and_scores(profile, args):
    method_variant = getattr(args, "method_variant", "baseline")
    if method_variant == "task4_acat_pairmargin_weight":
        if "high_acat_train_safe_hard_flag" not in profile.columns:
            raise ValueError("Task4 profile 缺少 high_acat_train_safe_hard_flag")
        flags = _bool_series(profile["high_acat_train_safe_hard_flag"])
        acat_score = _clip01(_numeric_task4_series(profile, "s_cat_v3"))
        hard_score = _clip01(_numeric_task4_series(profile, "train_safe_hard_proxy_score"))
        scores = (acat_score + hard_score) / 2.0
        return flags, scores
    if method_variant == "task4_acat_rsp_residual_pairmargin":
        if "high_acat_train_safe_hard_flag" not in profile.columns:
            raise ValueError("Task4 profile 缺少 high_acat_train_safe_hard_flag")
        if "RSP_group" not in profile.columns:
            raise ValueError("Task4 profile 缺少 RSP_group")
        acat_score = _clip01(_numeric_task4_series(profile, "s_cat_v3"))
        rsp_group = profile["RSP_group"].astype(str)
        group_mean = acat_score.groupby(rsp_group, dropna=False).transform("mean")
        residual = (acat_score - group_mean).clip(lower=0.0)
        max_residual = float(residual.max()) if len(residual) else 0.0
        residual_score = residual / max_residual if max_residual > 0 else residual
        flags = _bool_series(profile["high_acat_train_safe_hard_flag"]) & residual.gt(0)
        return flags, residual_score.clip(lower=0.0, upper=1.0)
    if method_variant == "task4_acat_hardonly_qmargin":
        if "train_safe_hard_proxy_high_flag" not in profile.columns:
            raise ValueError("Task4 profile 缺少 train_safe_hard_proxy_high_flag")
        flags = _task4_high_acat_flags(profile) & _bool_series(profile["train_safe_hard_proxy_high_flag"])
        scores = pd.Series(1.0, index=profile.index)
        return flags, scores
    if method_variant == "task4_highdetail_pairmargin":
        flags = _task4_highdetail_trainhard_flags(profile)
        acat_score = _clip01(_numeric_task4_series(profile, "s_cat_v3"))
        hard_score = _clip01(_numeric_task4_series(profile, "train_safe_hard_proxy_score"))
        scores = (acat_score + hard_score) / 2.0
        return flags, scores
    if method_variant == "task4_highdetail_pairmargin_shuffle":
        flags = _task4_highdetail_trainhard_flags(profile)
        acat_score = _clip01(_numeric_task4_series(profile, "s_cat_v3"))
        hard_score = _clip01(_numeric_task4_series(profile, "train_safe_hard_proxy_score"))
        scores = (acat_score + hard_score) / 2.0
        shuffled_flags, shuffled_scores = _task4_shuffle_by_split_and_detail(profile, flags, args, scores=scores)
        return _task4_high_detail_flags(profile) & shuffled_flags, _clip01(shuffled_scores)
    raise ValueError(f"unsupported Task4 pair-margin method_variant={method_variant}")


def build_task4_item_weights_from_profile(profile, item_serialize_dict, args, item_number=None):
    if not uses_task4_item_weights(args):
        return None
    if "raw_asin" not in profile.columns:
        raise ValueError("Task4 profile 缺少 raw_asin")
    if not item_serialize_dict:
        raise ValueError("item_serialize_dict must not be empty")
    if item_number is None:
        item_number = max(item_serialize_dict.values()) + 1
    alpha = float(getattr(args, "task4_loss_alpha", 0.5))
    if alpha <= 0:
        raise ValueError("task4_loss_alpha must be positive for Task4 variants")

    work = profile.copy().sort_values("raw_asin").reset_index(drop=True)
    method_variant = getattr(args, "method_variant", "baseline")
    if method_variant in M11R2_QBPR_WEIGHT_METHOD_VARIANTS:
        flags, scores = _m11r2_qbpr_flags_and_scores(work, method_variant)
    else:
        flags = _task4_variant_flags(work, args).astype(bool)
        scores = pd.Series(1.0, index=work.index, dtype=float)
    weights = torch.ones(int(item_number), dtype=torch.float32)
    for raw_asin, flag, score in zip(work["raw_asin"], flags.astype(bool), scores):
        serial_item = item_serialize_dict.get(raw_asin)
        if serial_item is None:
            continue
        if bool(flag):
            weights[int(serial_item)] = 1.0 + alpha * float(score)
    return weights / weights.mean().detach()


def build_m11r2_target_score_tensor_from_profile(profile, item_serialize_dict, item_number=None):
    if "raw_asin" not in profile.columns:
        raise ValueError("M11-R2 profile 缺少 raw_asin")
    required = ["m11r1_full_target_flag", "m11r1_full_target_loss_score"]
    missing = [column for column in required if column not in profile.columns]
    if missing:
        raise ValueError(f"M11-R2 profile 缺少 {missing}")
    if not item_serialize_dict:
        raise ValueError("item_serialize_dict must not be empty")
    if item_number is None:
        item_number = max(item_serialize_dict.values()) + 1

    work = profile.copy().sort_values("raw_asin").reset_index(drop=True)
    flags = _bool_series(work["m11r1_full_target_flag"])
    scores = _clip01(_numeric_task4_series(work, "m11r1_full_target_loss_score"))
    target_scores = torch.zeros(int(item_number), dtype=torch.float32)
    for raw_asin, flag, score in zip(work["raw_asin"], flags, scores):
        serial_item = item_serialize_dict.get(raw_asin)
        if serial_item is None or not bool(flag):
            continue
        target_scores[int(serial_item)] = float(score)
    return target_scores


def build_m11r2_focal_qbpr_weights(score_diff, target_scores, args):
    if target_scores is None:
        return None
    gamma = float(getattr(args, "m11r2_focal_gamma", 2.0))
    temperature = float(getattr(args, "m11r2_focal_temperature", 1.0))
    alpha = float(getattr(args, "task4_loss_alpha", 0.75))
    difficulty = torch.sigmoid(-score_diff / temperature).pow(gamma).detach()
    weights = 1.0 + alpha * target_scores.to(score_diff.dtype) * difficulty
    return weights / weights.mean().detach().clamp_min(1e-12)


def build_m11r2_curriculum_weights(full_weights, epoch_index, warmup_epochs):
    if full_weights is None:
        return None
    warmup_epochs = max(int(warmup_epochs), 1)
    progress = min(1.0, float(epoch_index + 1) / float(warmup_epochs))
    return 1.0 + progress * (full_weights - 1.0)


def cap_m11_residual_norm(residual, hidden, max_ratio):
    max_ratio = float(max_ratio)
    if max_ratio <= 0:
        raise ValueError("m11r3_residual_max_ratio must be positive")
    residual_norm = residual.norm(dim=1, keepdim=True)
    hidden_norm = hidden.detach().norm(dim=1, keepdim=True)
    allowed_norm = max_ratio * hidden_norm
    scale = torch.clamp(allowed_norm / residual_norm.clamp_min(1e-12), max=1.0)
    return residual * scale


def build_m11r3_neighbor_transfer_loss(residual, features, temperature=0.25):
    if residual is None or features is None:
        reference = residual if residual is not None else features
        if reference is None:
            return torch.tensor(0.0)
        return reference.new_tensor(0.0)
    if features.ndim != 2 or features.shape[1] != M11_FEATURE_WIDTH:
        raise ValueError(f"M11 neighbor features must have shape [batch,{M11_FEATURE_WIDTH}]")
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("m11r3_neighbor_temperature must be positive")

    target_mask = features[:, 0] >= 0.5
    non_target_mask = ~target_mask
    if not bool(target_mask.any()) or not bool(non_target_mask.any()):
        return residual.new_tensor(0.0)

    target_structure = features[target_mask, 1:].float()
    non_target_structure = features[non_target_mask, 1:].float()
    distances = torch.cdist(target_structure, non_target_structure, p=2)
    nearest_distance, nearest_index = distances.min(dim=1)
    target_residual = residual[target_mask].detach()
    neighbor_residual = residual[non_target_mask][nearest_index]
    confidence = torch.exp(-nearest_distance / temperature).detach()
    per_pair = F.smooth_l1_loss(neighbor_residual, target_residual, reduction="none").mean(dim=1)
    return (confidence * per_pair).sum()


def build_m11r4_relational_alignment_loss(q_v_c, features):
    if q_v_c is None or features is None:
        reference = q_v_c if q_v_c is not None else features
        if reference is None:
            return torch.tensor(0.0)
        return reference.new_tensor(0.0)
    if features.ndim != 2 or features.shape[1] != M11_FEATURE_WIDTH:
        raise ValueError(f"M11 relation features must have shape [batch,{M11_FEATURE_WIDTH}]")
    if q_v_c.ndim != 2 or q_v_c.shape[0] != features.shape[0]:
        raise ValueError("M11 relation q_v_c and features must share the batch dimension")

    target_mask = features[:, 0] >= 0.5
    if not bool(target_mask.any()) or q_v_c.shape[0] < 2:
        return q_v_c.new_tensor(0.0)

    structure = features[:, 1:].to(dtype=q_v_c.dtype)
    structure = structure - structure.mean(dim=0, keepdim=True)
    structure = F.normalize(structure, dim=1, eps=1e-6)
    representation = F.normalize(q_v_c, dim=1, eps=1e-6)
    structure_similarity = torch.matmul(structure[target_mask], structure.t()).detach()
    representation_similarity = torch.matmul(representation[target_mask], representation.t())

    coverage = 0.5 + 0.5 * features[:, 1].to(dtype=q_v_c.dtype).detach().clamp(0.0, 1.0)
    pair_weight = coverage.unsqueeze(0).expand_as(representation_similarity)
    per_pair = F.smooth_l1_loss(
        representation_similarity,
        structure_similarity,
        reduction="none",
    )
    normalized = (per_pair * pair_weight).sum() / pair_weight.sum().clamp_min(1e-12)
    return normalized * q_v_c.shape[0]


def build_m11r4_continuous_focal_weights(difficulty, features, args):
    if difficulty is None or features is None:
        return None
    if difficulty.ndim != 1:
        raise ValueError("M11-R4 focal difficulty must be one-dimensional")
    if features.ndim != 2 or features.shape != (difficulty.shape[0], M11_FEATURE_WIDTH):
        raise ValueError(f"M11-R4 focal features must have shape [batch,{M11_FEATURE_WIDTH}]")

    alpha = float(getattr(args, "m11r4_focal_alpha", 1.5))
    gamma = float(getattr(args, "m11r4_focal_gamma", 2.0))
    floor = float(getattr(args, "m11r4_focal_floor", 0.35))
    signal = features[:, 1].to(dtype=difficulty.dtype).detach().clamp(0.0, 1.0)
    coverage = floor + (1.0 - floor) * signal
    weights = 1.0 + alpha * coverage * difficulty.detach().clamp(0.0, 1.0).pow(gamma)
    return weights / weights.mean().detach().clamp_min(1e-12)


def cap_cicp_residual_norm(residual, hidden, max_ratio):
    max_ratio = float(max_ratio)
    if max_ratio <= 0 or max_ratio > 1:
        raise ValueError("cicp_residual_max_ratio must be in (0, 1]")
    residual_norm = residual.norm(dim=1, keepdim=True)
    allowed_norm = max_ratio * hidden.detach().norm(dim=1, keepdim=True)
    scale = torch.clamp(allowed_norm / residual_norm.clamp_min(1e-12), max=1.0)
    return residual * scale


def build_cicpr1_alignment_loss(q_v_c, item_embedding, cicp_features, epoch_index, args):
    if q_v_c is None or item_embedding is None or cicp_features is None:
        reference = q_v_c if q_v_c is not None else item_embedding
        if reference is None:
            return torch.tensor(0.0)
        return reference.new_tensor(0.0)
    if q_v_c.shape != item_embedding.shape:
        raise ValueError("CICP alignment q_v_c and item_embedding must have the same shape")
    if cicp_features.ndim != 2 or cicp_features.shape != (q_v_c.shape[0], CICP_FEATURE_WIDTH):
        raise ValueError(
            f"CICP alignment features must have shape [batch,{CICP_FEATURE_WIDTH}]"
        )
    warmup_epochs = int(getattr(args, "cicp_alignment_warmup_epochs", 20))
    if warmup_epochs <= 0:
        raise ValueError("cicp_alignment_warmup_epochs must be positive")
    progress = min(1.0, float(epoch_index + 1) / float(warmup_epochs))
    score = cicp_features[:, 0].detach().to(dtype=q_v_c.dtype).clamp(0.0, 1.0)
    item_teacher = item_embedding.detach()
    per_item = 1.0 - F.cosine_similarity(q_v_c, item_teacher, dim=1, eps=1e-6)
    weights = 0.25 + 0.75 * score
    return progress * (per_item * weights).sum()


def build_cicpr1_counterfactual_margin_loss(real_margin, shuffled_margin, cicp_features, args):
    if real_margin is None or shuffled_margin is None or cicp_features is None:
        reference = real_margin if real_margin is not None else shuffled_margin
        if reference is None:
            return torch.tensor(0.0)
        return reference.new_tensor(0.0)
    if real_margin.ndim != 1 or real_margin.shape != shuffled_margin.shape:
        raise ValueError("CICP counterfactual margins must be same-shape vectors")
    if cicp_features.ndim != 2 or cicp_features.shape != (
        real_margin.shape[0],
        CICP_FEATURE_WIDTH,
    ):
        raise ValueError(
            f"CICP counterfactual features must have shape [batch,{CICP_FEATURE_WIDTH}]"
        )
    margin = float(getattr(args, "cicp_counterfactual_margin", 0.05))
    if margin <= 0:
        raise ValueError("cicp_counterfactual_margin must be positive")
    score = cicp_features[:, 0].detach().to(dtype=real_margin.dtype).clamp(0.0, 1.0)
    target_gap = margin * score
    observed_gap = real_margin - shuffled_margin
    return (score * F.softplus(target_gap - observed_gap)).sum()


def build_cicpr2_score_distillation_loss(predicted_score, cicp_features):
    if predicted_score is None or cicp_features is None:
        reference = predicted_score if predicted_score is not None else cicp_features
        if reference is None:
            return torch.tensor(0.0)
        return reference.new_tensor(0.0)
    if predicted_score.ndim != 1:
        raise ValueError("CICP-R2 predicted scores must be one-dimensional")
    if cicp_features.ndim != 2 or cicp_features.shape != (
        predicted_score.shape[0],
        CICP_FEATURE_WIDTH,
    ):
        raise ValueError(
            f"CICP-R2 distillation features must have shape [batch,{CICP_FEATURE_WIDTH}]"
        )
    score = cicp_features[:, 0].detach().to(dtype=predicted_score.dtype).clamp(0.0, 1.0)
    return F.smooth_l1_loss(predicted_score, score, reduction="sum")


def build_cicpr2_ordinal_counterfactual_loss(
    real_embedding,
    shuffled_embedding,
    item_embedding,
    cicp_features,
    args,
):
    if real_embedding is None or shuffled_embedding is None or item_embedding is None:
        reference = real_embedding if real_embedding is not None else shuffled_embedding
        if reference is None:
            return torch.tensor(0.0)
        return reference.new_tensor(0.0)
    if real_embedding.ndim != 2 or shuffled_embedding.shape != real_embedding.shape:
        raise ValueError("CICP-R2 real and shuffled embeddings must have the same 2D shape")
    if item_embedding.shape != real_embedding.shape:
        raise ValueError("CICP-R2 item teacher must match generated embedding shape")
    if cicp_features is None or cicp_features.shape != (
        real_embedding.shape[0],
        CICP_FEATURE_WIDTH,
    ):
        raise ValueError(
            f"CICP-R2 ordinal features must have shape [batch,{CICP_FEATURE_WIDTH}]"
        )
    pair_count = real_embedding.shape[0] // 2
    if pair_count == 0:
        return real_embedding.new_tensor(0.0)

    teacher = item_embedding.detach()
    real_similarity = F.cosine_similarity(real_embedding, teacher, dim=1, eps=1e-6)
    shuffled_similarity = F.cosine_similarity(shuffled_embedding, teacher, dim=1, eps=1e-6)
    observed_increment = real_similarity - shuffled_similarity
    score = cicp_features[:, 0].detach().to(dtype=observed_increment.dtype).clamp(0.0, 1.0)
    order = torch.argsort(score)
    low_index = order[:pair_count]
    high_index = order[-pair_count:]
    score_gap = (score[high_index] - score[low_index]).clamp_min(1e-6)
    increment_gap = observed_increment[high_index] - observed_increment[low_index]
    margin = float(getattr(args, "cicpr2_ordinal_margin", 0.02))
    if margin <= 0:
        raise ValueError("cicpr2_ordinal_margin must be positive")
    per_pair = F.softplus(margin - increment_gap)
    normalized = (per_pair * score_gap).sum() / score_gap.sum().clamp_min(1e-12)
    return normalized * real_embedding.shape[0]


def build_task4_pair_margin_targets_from_profile(profile, item_serialize_dict, args, item_number=None):
    if not uses_task4_pair_margin(args):
        return None
    if "raw_asin" not in profile.columns:
        raise ValueError("Task4 profile 缺少 raw_asin")
    if not item_serialize_dict:
        raise ValueError("item_serialize_dict must not be empty")
    if item_number is None:
        item_number = max(item_serialize_dict.values()) + 1
    alpha = float(getattr(args, "task4_loss_alpha", 0.5))
    if alpha <= 0:
        raise ValueError("task4_loss_alpha must be positive for Task4 variants")
    base_margin = float(getattr(args, "task4_pair_margin", 0.2))
    if base_margin <= 0:
        raise ValueError("task4_pair_margin must be positive for Task4 pair-margin variants")

    work = profile.copy().sort_values("raw_asin").reset_index(drop=True)
    flags, scores = _task4_pair_margin_flags_and_scores(work, args)
    scores = _clip01(scores)
    loss_weight = torch.zeros(int(item_number), dtype=torch.float32)
    margin = torch.full((int(item_number),), base_margin, dtype=torch.float32)
    for raw_asin, flag, score in zip(work["raw_asin"], flags.astype(bool), scores):
        serial_item = item_serialize_dict.get(raw_asin)
        if serial_item is None:
            continue
        serial_item = int(serial_item)
        score = float(score)
        if bool(flag):
            loss_weight[serial_item] = alpha * (0.5 + 0.5 * score)
            margin[serial_item] = base_margin * (1.0 + score)
    return {"loss_weight": loss_weight, "margin": margin}


def _m11_target_scores(profile):
    if "m11_target_score" in profile.columns:
        return _clip01(_numeric_task4_series(profile, "m11_target_score"))
    acat_score = _clip01(_numeric_task4_series(profile, "s_cat_v3"))
    hard_score = _clip01(_numeric_task4_series(profile, "train_safe_hard_proxy_score"))
    if "RSP_score" in profile.columns:
        rsp_inverse = (1.0 - _clip01(_numeric_task4_series(profile, "RSP_score"))).clip(lower=0.0, upper=1.0)
    else:
        rsp_inverse = pd.Series(0.5, index=profile.index)
    return _clip01((0.45 * acat_score) + (0.35 * hard_score) + (0.20 * rsp_inverse))


def _m11_neighbor_support_flags(profile):
    missing = [
        column
        for column in ["category_neighbor_mismatch_proxy_high_flag", "support_tail_proxy_high_flag"]
        if column not in profile.columns
    ]
    if missing:
        raise ValueError(f"M11 profile 缺少 {missing}")
    return (
        _bool_series(profile["category_neighbor_mismatch_proxy_high_flag"])
        & _bool_series(profile["support_tail_proxy_high_flag"])
    )


def _m11_low_rsp_flags(profile):
    if "RSP_group" not in profile.columns:
        raise ValueError("M11 profile 缺少 RSP_group")
    return ~profile["RSP_group"].astype(str).eq("RSP_high")


def _m11_target_flags(profile):
    if "m11_high_acat_low_rsp_neighbor_support_flag" in profile.columns:
        return _bool_series(profile["m11_high_acat_low_rsp_neighbor_support_flag"])
    return _task4_high_acat_flags(profile) & _m11_low_rsp_flags(profile) & _m11_neighbor_support_flags(profile)


def _m11_lowrsp_matched_control_flags(profile):
    return (~_task4_high_acat_flags(profile)) & _m11_low_rsp_flags(profile) & _m11_neighbor_support_flags(profile)


def _m11_rsp_control_flags(profile):
    if "RSP_group" not in profile.columns:
        raise ValueError("M11 profile 缺少 RSP_group")
    return profile["RSP_group"].astype(str).eq("RSP_high") & _m11_neighbor_support_flags(profile)


def _task4_competitor_pair_flags_and_scores(profile, args):
    method_variant = getattr(args, "method_variant", "baseline")
    if method_variant in {"task4_competitor_pair", "task4_boundary_competitor_pair"}:
        flags = _task4_highdetail_trainhard_flags(profile)
        acat_score = _clip01(_numeric_task4_series(profile, "s_cat_v3"))
        hard_score = _clip01(_numeric_task4_series(profile, "train_safe_hard_proxy_score"))
        return flags, (acat_score + hard_score) / 2.0
    if method_variant in {"task4_competitor_pair_shuffle", "task4_boundary_competitor_pair_shuffle"}:
        flags = _task4_highdetail_trainhard_flags(profile)
        acat_score = _clip01(_numeric_task4_series(profile, "s_cat_v3"))
        hard_score = _clip01(_numeric_task4_series(profile, "train_safe_hard_proxy_score"))
        scores = (acat_score + hard_score) / 2.0
        shuffled_flags, shuffled_scores = _task4_shuffle_by_split_and_detail(profile, flags, args, scores=scores)
        return _task4_high_detail_flags(profile) & shuffled_flags, _clip01(shuffled_scores)
    if method_variant in {"task4_competitor_pair_rsp_control", "task4_boundary_competitor_pair_rsp_control"}:
        if "RSP_group" not in profile.columns:
            raise ValueError("Task4 profile 缺少 RSP_group")
        flags = _task4_high_detail_flags(profile) & profile["RSP_group"].astype(str).eq("RSP_high")
        if "RSP_score" in profile.columns:
            scores = _clip01(_numeric_task4_series(profile, "RSP_score"))
        else:
            scores = pd.Series(1.0, index=profile.index)
        return flags, scores
    if method_variant in {"task4_competitor_pair_acat_control", "task4_boundary_competitor_pair_acat_control"}:
        flags = _task4_high_detail_flags(profile) & _task4_high_acat_flags(profile)
        return flags, _clip01(_numeric_task4_series(profile, "s_cat_v3"))
    if method_variant == "m11_target_competitor_pair":
        return _m11_target_flags(profile), _m11_target_scores(profile)
    if method_variant == "m11_target_competitor_pair_shuffle":
        flags = _m11_target_flags(profile)
        scores = _m11_target_scores(profile)
        shuffled_flags, shuffled_scores = _task4_shuffle_by_split_and_detail(profile, flags, args, scores=scores)
        return shuffled_flags.astype(bool), _clip01(shuffled_scores)
    if method_variant == "m11_target_competitor_pair_lowrsp_control":
        return _m11_lowrsp_matched_control_flags(profile), _m11_target_scores(profile)
    if method_variant == "m11_target_competitor_pair_rsp_control":
        return _m11_rsp_control_flags(profile), _m11_target_scores(profile)
    m11r1_columns = {
        "m11r1_full_target_competitor_pair": (
            "m11r1_full_target_flag",
            "m11r1_full_target_loss_score",
        ),
        "m11r1_popmatch_competitor_pair_control": (
            "m11r1_popmatch_control_flag",
            "m11r1_popmatch_control_loss_score",
        ),
        "m11r1_lowacat_competitor_pair_control": (
            "m11r1_lowacat_control_flag",
            "m11r1_lowacat_control_loss_score",
        ),
    }
    if method_variant in m11r1_columns:
        flag_column, score_column = m11r1_columns[method_variant]
        missing = [column for column in [flag_column, score_column] if column not in profile.columns]
        if missing:
            raise ValueError(f"M11-R1 profile 缺少 {missing}")
        return _bool_series(profile[flag_column]), _clip01(_numeric_task4_series(profile, score_column))
    raise ValueError(f"unsupported Task4 competitor-pair method_variant={method_variant}")


def build_task4_competitor_pair_targets_from_profile(profile, item_serialize_dict, args, item_number=None):
    if not uses_task4_competitor_pair(args):
        return None
    if "raw_asin" not in profile.columns:
        raise ValueError("Task4 profile 缺少 raw_asin")
    if not item_serialize_dict:
        raise ValueError("item_serialize_dict must not be empty")
    if item_number is None:
        item_number = max(item_serialize_dict.values()) + 1
    alpha = float(getattr(args, "task4_competitor_alpha", 0.25))
    if alpha <= 0:
        raise ValueError("task4_competitor_alpha must be positive for Task4 competitor-pair variants")
    base_margin = float(getattr(args, "task4_competitor_margin", 0.1))
    if base_margin <= 0:
        raise ValueError("task4_competitor_margin must be positive for Task4 competitor-pair variants")

    work = profile.copy().sort_values("raw_asin").reset_index(drop=True)
    flags, scores = _task4_competitor_pair_flags_and_scores(work, args)
    scores = _clip01(scores)
    loss_weight = torch.zeros(int(item_number), dtype=torch.float32)
    margin = torch.full((int(item_number),), base_margin, dtype=torch.float32)
    for raw_asin, flag, score in zip(work["raw_asin"], flags.astype(bool), scores):
        serial_item = item_serialize_dict.get(raw_asin)
        if serial_item is None:
            continue
        serial_item = int(serial_item)
        score = float(score)
        if bool(flag):
            loss_weight[serial_item] = alpha * (0.5 + 0.5 * score)
            margin[serial_item] = base_margin * (1.0 + score)
    return {"loss_weight": loss_weight, "margin": margin}


def load_task4_item_weights(item_serialize_dict, args, item_number=None):
    if not uses_task4_item_weights(args):
        return None
    profile_path = getattr(args, "task4_profile_path", "")
    if not profile_path:
        raise ValueError("task4_profile_path is required for Task4 variants")
    profile = pd.read_csv(profile_path)
    return build_task4_item_weights_from_profile(profile, item_serialize_dict, args, item_number=item_number)


def load_m11r2_target_score_tensor(item_serialize_dict, args, item_number=None):
    if not uses_m11r2_focal_qbpr(args):
        return None
    profile_path = getattr(args, "task4_profile_path", "")
    if not profile_path:
        raise ValueError("task4_profile_path is required for M11-R2 focal qBPR")
    profile = pd.read_csv(profile_path, dtype={"raw_asin": str}, low_memory=False)
    return build_m11r2_target_score_tensor_from_profile(
        profile,
        item_serialize_dict,
        item_number=item_number,
    )


def load_m11r2_feature_tensor(item_serialize_dict, args, item_number=None):
    if not uses_m11r2_feature_fusion(args):
        return None
    profile_path = getattr(args, "task4_profile_path", "")
    if not profile_path:
        raise ValueError("task4_profile_path is required for M11-R2 feature fusion")
    return load_m11_feature_tensor(
        profile_path,
        item_serialize_dict,
        item_number=item_number,
        feature_mode=resolve_m11_feature_mode(args),
        reject_evaluation_columns=(
            getattr(args, "method_variant", "baseline") in M11R4_FEATURE_METHOD_VARIANTS
        ),
    )


def load_cicpr1_feature_tensor(item_serialize_dict, args, item_number=None):
    if not uses_cicp_features(args):
        return None
    profile_path = getattr(args, "cicp_profile_path", "")
    if not profile_path:
        raise ValueError("cicp_profile_path is required for CICP variants")
    return load_cicp_feature_tensor(
        profile_path,
        item_serialize_dict,
        item_number=item_number,
        reject_evaluation_columns=True,
    )


def load_task4_pair_margin_targets(item_serialize_dict, args, item_number=None):
    if not uses_task4_pair_margin(args):
        return None
    profile_path = getattr(args, "task4_profile_path", "")
    if not profile_path:
        raise ValueError("task4_profile_path is required for Task4 variants")
    profile = pd.read_csv(profile_path)
    return build_task4_pair_margin_targets_from_profile(profile, item_serialize_dict, args, item_number=item_number)


def load_task4_competitor_pair_targets(item_serialize_dict, args, item_number=None):
    if not uses_task4_competitor_pair(args):
        return None
    profile_path = getattr(args, "task4_profile_path", "")
    if not profile_path:
        raise ValueError("task4_profile_path is required for Task4 variants")
    profile = pd.read_csv(profile_path)
    return build_task4_competitor_pair_targets_from_profile(profile, item_serialize_dict, args, item_number=item_number)


def _lookup_serial_id(mapping, raw_value):
    if pd.isna(raw_value):
        return None
    text = str(raw_value).strip()
    candidates = [raw_value, text]
    if text:
        try:
            int_value = int(text)
            candidates.extend([int_value, np.int64(int_value)])
        except ValueError:
            pass
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return None


def load_task4_boundary_competitors(item_serialize_dict, user_serialize_dict, args, item_number=None):
    if not uses_task4_boundary_competitor_pair(args):
        return None
    cache_path = str(getattr(args, "task4_boundary_competitor_cache_path", "")).strip()
    if not cache_path:
        raise ValueError("task4_boundary_competitor_cache_path is required for Task4 boundary competitor variants")
    cache = pd.read_csv(cache_path, dtype={"boundary_competitor_user": str}, low_memory=False)
    if "raw_asin" not in cache.columns:
        raise ValueError("Task4 boundary competitor cache 缺少 raw_asin")
    if "boundary_competitor_user" not in cache.columns and "boundary_competitor_serial_user" not in cache.columns:
        raise ValueError("Task4 boundary competitor cache 缺少 boundary_competitor_user 或 boundary_competitor_serial_user")
    if item_number is None:
        item_number = max(item_serialize_dict.values()) + 1
    boundary_users = torch.full((int(item_number),), -1, dtype=torch.long)
    loaded = 0
    for _, row in cache.iterrows():
        serial_item = _lookup_serial_id(item_serialize_dict, row["raw_asin"])
        if serial_item is None:
            continue
        serial_user = None
        if "boundary_competitor_user" in cache.columns and pd.notna(row.get("boundary_competitor_user")):
            serial_user = _lookup_serial_id(user_serialize_dict, row["boundary_competitor_user"])
        if serial_user is None and "boundary_competitor_serial_user" in cache.columns and pd.notna(row.get("boundary_competitor_serial_user")):
            serial_user = int(row["boundary_competitor_serial_user"])
        if serial_user is None:
            continue
        serial_user = int(serial_user)
        if serial_user < 0 or serial_user >= len(user_serialize_dict):
            raise ValueError(f"boundary competitor user id out of range: {serial_user}")
        boundary_users[int(serial_item)] = serial_user
        loaded += 1
    if loaded == 0:
        raise ValueError(f"Task4 boundary competitor cache produced no mapped competitors: {cache_path}")
    return boundary_users


def resolve_task4_competitor_user_ids(item, neg_user, task4_boundary_competitors):
    if task4_boundary_competitors is None:
        return neg_user
    boundary_user = task4_boundary_competitors[item]
    return torch.where(boundary_user >= 0, boundary_user, neg_user)


def weighted_sum(per_item_loss, weights, enabled):
    if weights is None or enabled is False:
        return per_item_loss.sum()
    return (per_item_loss * weights).sum()


def task4_pair_margin_loss(score_diff, targets):
    if targets is None:
        return score_diff.new_tensor(0.0)
    loss_weight = targets["loss_weight"].to(device=score_diff.device, dtype=score_diff.dtype)
    margin = targets["margin"].to(device=score_diff.device, dtype=score_diff.dtype)
    return (torch.relu(margin - score_diff) * loss_weight).sum()


def task4_competitor_pair_loss(score_diff, targets):
    if targets is None:
        return score_diff.new_tensor(0.0)
    loss_weight = targets["loss_weight"].to(device=score_diff.device, dtype=score_diff.dtype)
    margin = targets["margin"].to(device=score_diff.device, dtype=score_diff.dtype)
    return (F.softplus(margin - score_diff) * loss_weight).sum()


def validate_method_args(args):
    method_variant = getattr(args, "method_variant", "baseline")
    reweight_flags = [
        getattr(args, "reweight_q_bpr", False),
        getattr(args, "reweight_self_contrast", False),
        getattr(args, "reweight_contrast", False),
    ]
    if method_variant != "weak_q_reweight" and any(reweight_flags):
        raise ValueError("reweight flags can only be used with method_variant=weak_q_reweight")
    category_conf_variants = {"category_conf_input", "category_conf_fusion_gate"}
    if method_variant in category_conf_variants:
        if int(getattr(args, "category_conf_dim", 16)) <= 0:
            raise ValueError("category_conf_dim must be positive for category confidence variants")
        if int(getattr(args, "category_conf_max_count", 5)) <= 0:
            raise ValueError("category_conf_max_count must be positive for category confidence variants")
    if method_variant == "category_conf_fusion_gate":
        category_gate_scale = float(getattr(args, "category_gate_scale", 0.5))
        if category_gate_scale <= 0 or category_gate_scale > 1:
            raise ValueError("category_gate_scale must be in (0, 1] for category_conf_fusion_gate")
    if method_variant == "adaptive_conf_qbpr":
        if float(getattr(args, "adaptive_loss_alpha", 1.0)) <= 0:
            raise ValueError("adaptive_loss_alpha must be positive for adaptive_conf_qbpr")
        if int(getattr(args, "adaptive_history_max_count", 20)) <= 0:
            raise ValueError("adaptive_history_max_count must be positive for adaptive_conf_qbpr")
    if method_variant in TASK4_METHOD_VARIANTS:
        if not str(getattr(args, "task4_profile_path", "")).strip():
            raise ValueError("task4_profile_path is required for Task4 variants")
        if float(getattr(args, "task4_loss_alpha", 0.5)) <= 0:
            raise ValueError("task4_loss_alpha must be positive for Task4 variants")
    if method_variant in TASK4_PAIR_MARGIN_METHOD_VARIANTS:
        if float(getattr(args, "task4_pair_margin", 0.2)) <= 0:
            raise ValueError("task4_pair_margin must be positive for Task4 pair-margin variants")
    if method_variant in TASK4_COMPETITOR_PAIR_METHOD_VARIANTS:
        if float(getattr(args, "task4_competitor_alpha", 0.25)) <= 0:
            raise ValueError("task4_competitor_alpha must be positive for Task4 competitor-pair variants")
        if float(getattr(args, "task4_competitor_margin", 0.1)) <= 0:
            raise ValueError("task4_competitor_margin must be positive for Task4 competitor-pair variants")
        if int(getattr(args, "task4_competitor_k", 20)) <= 0:
            raise ValueError("task4_competitor_k must be positive for Task4 competitor-pair variants")
    if method_variant in TASK4_BOUNDARY_COMPETITOR_PAIR_METHOD_VARIANTS:
        if not str(getattr(args, "task4_boundary_competitor_cache_path", "")).strip():
            raise ValueError("task4_boundary_competitor_cache_path is required for Task4 boundary competitor variants")
    if method_variant in M11R2_FOCAL_METHOD_VARIANTS:
        if float(getattr(args, "m11r2_focal_gamma", 2.0)) <= 0:
            raise ValueError("m11r2_focal_gamma must be positive")
        if float(getattr(args, "m11r2_focal_temperature", 1.0)) <= 0:
            raise ValueError("m11r2_focal_temperature must be positive")
    if method_variant == "m11r2_qbpr_curriculum":
        if int(getattr(args, "m11r2_curriculum_warmup_epochs", 20)) <= 0:
            raise ValueError("m11r2_curriculum_warmup_epochs must be positive")
    if method_variant in M11R2_FEATURE_METHOD_VARIANTS:
        if int(getattr(args, "m11r2_feature_dim", 16)) <= 0:
            raise ValueError("m11r2_feature_dim must be positive")
    if method_variant in M11R3_NORM_CAPPED_METHOD_VARIANTS:
        max_ratio = float(getattr(args, "m11r3_residual_max_ratio", 0.15))
        if max_ratio <= 0 or max_ratio > 1:
            raise ValueError("m11r3_residual_max_ratio must be in (0, 1]")
    if method_variant in M11R3_NEIGHBOR_TRANSFER_METHOD_VARIANTS:
        if float(getattr(args, "m11r3_neighbor_loss_weight", 0.1)) <= 0:
            raise ValueError("m11r3_neighbor_loss_weight must be positive")
        if float(getattr(args, "m11r3_neighbor_temperature", 0.25)) <= 0:
            raise ValueError("m11r3_neighbor_temperature must be positive")
    if method_variant in M11R3_FILM_METHOD_VARIANTS:
        film_strength = float(getattr(args, "m11r3_film_strength", 0.1))
        if film_strength <= 0 or film_strength > 1:
            raise ValueError("m11r3_film_strength must be in (0, 1]")
    if method_variant in M11R4_PROTECTED_EXPERT_METHOD_VARIANTS:
        strength = float(getattr(args, "m11r4_expert_film_strength", 0.2))
        if strength <= 0 or strength > 1:
            raise ValueError("m11r4_expert_film_strength must be in (0, 1]")
    if method_variant in M11R4_CONTINUOUS_FUSION_METHOD_VARIANTS:
        strength = float(getattr(args, "m11r4_fusion_strength", 0.25))
        if strength <= 0 or strength > 1:
            raise ValueError("m11r4_fusion_strength must be in (0, 1]")
    if method_variant in M11R4_RELATIONAL_ALIGNMENT_METHOD_VARIANTS:
        if float(getattr(args, "m11r4_relation_loss_weight", 0.05)) <= 0:
            raise ValueError("m11r4_relation_loss_weight must be positive")
    if method_variant in M11R4_CONTINUOUS_FOCAL_METHOD_VARIANTS:
        if float(getattr(args, "m11r4_focal_alpha", 1.5)) <= 0:
            raise ValueError("m11r4_focal_alpha must be positive")
        if float(getattr(args, "m11r4_focal_gamma", 2.0)) <= 0:
            raise ValueError("m11r4_focal_gamma must be positive")
        if float(getattr(args, "m11r4_focal_temperature", 0.5)) <= 0:
            raise ValueError("m11r4_focal_temperature must be positive")
        floor = float(getattr(args, "m11r4_focal_floor", 0.35))
        if floor <= 0 or floor > 1:
            raise ValueError("m11r4_focal_floor must be in (0, 1]")
    if method_variant in CICP_METHOD_VARIANTS:
        if not str(getattr(args, "cicp_profile_path", "")).strip():
            raise ValueError("cicp_profile_path is required for CICP variants")
    if method_variant in CICPR1_E4_RESIDUAL_METHOD_VARIANTS:
        if int(getattr(args, "cicp_feature_dim", 16)) <= 0:
            raise ValueError("cicp_feature_dim must be positive")
        ratio = float(getattr(args, "cicp_residual_max_ratio", 0.15))
        if ratio <= 0 or ratio > 1:
            raise ValueError("cicp_residual_max_ratio must be in (0, 1]")
    if method_variant in CICPR1_MODALITY_ROUTING_METHOD_VARIANTS:
        strength = float(getattr(args, "cicp_modality_strength", 0.25))
        if strength <= 0 or strength > 1:
            raise ValueError("cicp_modality_strength must be in (0, 1]")
    if method_variant in CICPR1_CATEGORY_EXPERT_METHOD_VARIANTS:
        strength = float(getattr(args, "cicp_expert_strength", 0.20))
        if strength <= 0 or strength > 1:
            raise ValueError("cicp_expert_strength must be in (0, 1]")
    if method_variant in CICPR1_ALIGNMENT_METHOD_VARIANTS:
        if float(getattr(args, "cicp_alignment_weight", 0.05)) <= 0:
            raise ValueError("cicp_alignment_weight must be positive")
        if int(getattr(args, "cicp_alignment_warmup_epochs", 20)) <= 0:
            raise ValueError("cicp_alignment_warmup_epochs must be positive")
    if method_variant in CICPR1_COUNTERFACTUAL_METHOD_VARIANTS:
        if float(getattr(args, "cicp_counterfactual_weight", 0.05)) <= 0:
            raise ValueError("cicp_counterfactual_weight must be positive")
        if float(getattr(args, "cicp_counterfactual_margin", 0.05)) <= 0:
            raise ValueError("cicp_counterfactual_margin must be positive")
    if method_variant in CICPR1_ADAPTIVE_ATTENTION_METHOD_VARIANTS:
        strength = float(getattr(args, "cicp_attention_strength", 0.50))
        if strength <= 0 or strength > 1:
            raise ValueError("cicp_attention_strength must be in (0, 1]")
    if method_variant in CICPR2_CONTENT_DIRECTION_RESIDUAL_METHOD_VARIANTS:
        ratio = float(getattr(args, "cicpr2_residual_max_ratio", 0.15))
        if ratio <= 0 or ratio > 1:
            raise ValueError("cicpr2_residual_max_ratio must be in (0, 1]")
    if method_variant in CICPR2_CATEGORY_INCREMENT_METHOD_VARIANTS:
        strength = float(getattr(args, "cicpr2_increment_strength", 0.50))
        if strength <= 0 or strength > 1:
            raise ValueError("cicpr2_increment_strength must be in (0, 1]")
    if method_variant in CICPR2_CROSS_MODAL_ATTENTION_METHOD_VARIANTS:
        strength = float(getattr(args, "cicpr2_cross_attention_strength", 0.50))
        if strength <= 0 or strength > 1:
            raise ValueError("cicpr2_cross_attention_strength must be in (0, 1]")
        if float(getattr(args, "cicpr2_cross_attention_temperature", 0.25)) <= 0:
            raise ValueError("cicpr2_cross_attention_temperature must be positive")
    if method_variant in CICPR2_SCORE_DISTILLATION_METHOD_VARIANTS:
        if float(getattr(args, "cicpr2_distillation_weight", 0.05)) <= 0:
            raise ValueError("cicpr2_distillation_weight must be positive")
    if method_variant in CICPR2_ORDINAL_COUNTERFACTUAL_METHOD_VARIANTS:
        if float(getattr(args, "cicpr2_ordinal_weight", 0.05)) <= 0:
            raise ValueError("cicpr2_ordinal_weight must be positive")
        if float(getattr(args, "cicpr2_ordinal_margin", 0.02)) <= 0:
            raise ValueError("cicpr2_ordinal_margin must be positive")
    if method_variant in CICPR2_RELIABILITY_DROPOUT_METHOD_VARIANTS:
        dropout = float(getattr(args, "cicpr2_category_dropout_max", 0.50))
        if dropout <= 0 or dropout >= 1:
            raise ValueError("cicpr2_category_dropout_max must be in (0, 1)")


def scalar_text(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    return str(value)


def training_result_header():
    return (
        "checkpoint_index,epoch,batch,total_batches,elapsed_s,"
        "loss,contrast_sum,hr@5,hr@10,hr@20,ndcg@5,ndcg@10,ndcg@20\n"
    )


def build_run_config(args, model):
    method_variant = getattr(args, "method_variant", "baseline")
    return {
        "method_variant": method_variant,
        "category_conf_dim": int(getattr(args, "category_conf_dim", 0)),
        "category_conf_max_count": int(getattr(args, "category_conf_max_count", 0)),
        "category_bin_count": int(getattr(model, "category_bin_count", 0)),
        "category_gate_scale": float(getattr(args, "category_gate_scale", 0.0)),
        "category_fusion_gate_output_dim": int(
            model.category_fusion_gate.out_features if hasattr(model, "category_fusion_gate") else 0
        ),
        "gen_layer1_input_dim": int(model.gen_layer1.in_features),
        "gen_layer1_output_dim": int(model.gen_layer1.out_features),
        "weak_cat_threshold": int(getattr(args, "weak_cat_threshold", 3)),
        "weak_loss_alpha": float(getattr(args, "weak_loss_alpha", 0.0)),
        "adaptive_loss_alpha": float(getattr(args, "adaptive_loss_alpha", 0.0)),
        "adaptive_history_max_count": int(getattr(args, "adaptive_history_max_count", 0)),
        "task4_profile_path": str(getattr(args, "task4_profile_path", "")),
        "task4_loss_alpha": float(getattr(args, "task4_loss_alpha", 0.0)),
        "task4_shuffle_seed": int(getattr(args, "task4_shuffle_seed", 43)),
        "task4_disable_q_bpr_weight": bool(getattr(args, "task4_disable_q_bpr_weight", False)),
        "task4_disable_self_contrast_weight": bool(getattr(args, "task4_disable_self_contrast_weight", False)),
        "task4_reweight_contrast": bool(getattr(args, "task4_reweight_contrast", False)),
        "task4_pair_margin": float(getattr(args, "task4_pair_margin", 0.0)),
        "task4_competitor_alpha": float(getattr(args, "task4_competitor_alpha", 0.0)),
        "task4_competitor_margin": float(getattr(args, "task4_competitor_margin", 0.0)),
        "task4_competitor_k": int(getattr(args, "task4_competitor_k", 0)),
        "task4_boundary_competitor_cache_path": str(getattr(args, "task4_boundary_competitor_cache_path", "")),
        "m11r2_focal_gamma": float(getattr(args, "m11r2_focal_gamma", 0.0)),
        "m11r2_focal_temperature": float(getattr(args, "m11r2_focal_temperature", 0.0)),
        "m11r2_curriculum_warmup_epochs": int(getattr(args, "m11r2_curriculum_warmup_epochs", 0)),
        "m11r2_feature_dim": int(getattr(args, "m11r2_feature_dim", 0)),
        "m11r2_feature_input_width": int(M11_FEATURE_WIDTH if uses_m11r2_feature_fusion(args) else 0),
        "m11_feature_mode": resolve_m11_feature_mode(args) if uses_m11r2_feature_fusion(args) else "none",
        "m11r3_residual_max_ratio": float(getattr(args, "m11r3_residual_max_ratio", 0.0)),
        "m11r3_neighbor_loss_weight": float(getattr(args, "m11r3_neighbor_loss_weight", 0.0)),
        "m11r3_neighbor_temperature": float(getattr(args, "m11r3_neighbor_temperature", 0.0)),
        "m11r3_film_strength": float(getattr(args, "m11r3_film_strength", 0.0)),
        "m11r4_expert_film_strength": float(getattr(args, "m11r4_expert_film_strength", 0.0)),
        "m11r4_fusion_strength": float(getattr(args, "m11r4_fusion_strength", 0.0)),
        "m11r4_relation_loss_weight": float(getattr(args, "m11r4_relation_loss_weight", 0.0)),
        "m11r4_focal_alpha": float(getattr(args, "m11r4_focal_alpha", 0.0)),
        "m11r4_focal_gamma": float(getattr(args, "m11r4_focal_gamma", 0.0)),
        "m11r4_focal_temperature": float(getattr(args, "m11r4_focal_temperature", 0.0)),
        "m11r4_focal_floor": float(getattr(args, "m11r4_focal_floor", 0.0)),
        "cicp_profile_path": str(getattr(args, "cicp_profile_path", "")),
        "cicp_feature_input_width": int(CICP_FEATURE_WIDTH if uses_cicp_features(args) else 0),
        "cicp_feature_dim": int(getattr(args, "cicp_feature_dim", 0)),
        "cicp_residual_max_ratio": float(getattr(args, "cicp_residual_max_ratio", 0.0)),
        "cicp_modality_strength": float(getattr(args, "cicp_modality_strength", 0.0)),
        "cicp_expert_strength": float(getattr(args, "cicp_expert_strength", 0.0)),
        "cicp_alignment_weight": float(getattr(args, "cicp_alignment_weight", 0.0)),
        "cicp_alignment_warmup_epochs": int(getattr(args, "cicp_alignment_warmup_epochs", 0)),
        "cicp_counterfactual_weight": float(getattr(args, "cicp_counterfactual_weight", 0.0)),
        "cicp_counterfactual_margin": float(getattr(args, "cicp_counterfactual_margin", 0.0)),
        "cicp_attention_strength": float(getattr(args, "cicp_attention_strength", 0.0)),
        "cicp_e4_residual_is_unique": method_variant in CICPR1_E4_RESIDUAL_METHOD_VARIANTS,
        "cicpr2_residual_max_ratio": float(getattr(args, "cicpr2_residual_max_ratio", 0.0)),
        "cicpr2_increment_strength": float(getattr(args, "cicpr2_increment_strength", 0.0)),
        "cicpr2_cross_attention_strength": float(
            getattr(args, "cicpr2_cross_attention_strength", 0.0)
        ),
        "cicpr2_cross_attention_temperature": float(
            getattr(args, "cicpr2_cross_attention_temperature", 0.0)
        ),
        "cicpr2_distillation_weight": float(
            getattr(args, "cicpr2_distillation_weight", 0.0)
        ),
        "cicpr2_ordinal_weight": float(getattr(args, "cicpr2_ordinal_weight", 0.0)),
        "cicpr2_ordinal_margin": float(getattr(args, "cicpr2_ordinal_margin", 0.0)),
        "cicpr2_category_dropout_max": float(
            getattr(args, "cicpr2_category_dropout_max", 0.0)
        ),
        "cicpr2_e4_style_residual_is_unique": (
            method_variant in CICPR2_CONTENT_DIRECTION_RESIDUAL_METHOD_VARIANTS
        ),
        "cicpr2_independent_signal_width": 1 if method_variant in CICPR2_METHOD_VARIANTS else 0,
        "training_input_uses_validation_item_metrics": False,
        "training_input_uses_test_item_metrics": False,
        "seed": int(getattr(args, "seed", -1)),
        "num_workers": int(getattr(args, "num_workers", 0)),
        "batch_size": int(getattr(args, "batch_size", 0)),
        "save_batch_time": int(getattr(args, "save_batch_time", 0)),
        "validate_batch_size": int(getattr(args, "validate_batch_size", 0)),
        "negative_sampling_mode": str(getattr(args, "negative_sampling_mode", "")),
        "negative_sampling_cache_size": int(getattr(args, "negative_sampling_cache_size", 0)),
    }


def write_run_config(args, model, model_save_dir):
    with open(os.path.join(model_save_dir, "run_config.json"), "w") as f:
        json.dump(build_run_config(args, model), f, indent=2, sort_keys=True)


def build_training_result_row(
    checkpoint_index,
    epoch,
    batch,
    total_batches,
    elapsed_s,
    loss,
    contrast_sum,
    metrics,
):
    values = [
        checkpoint_index,
        epoch,
        batch,
        total_batches,
        elapsed_s,
        scalar_text(loss),
        scalar_text(contrast_sum),
        *[scalar_text(metric) for metric in metrics],
    ]
    return ",".join(str(value) for value in values) + "\n"


def format_training_progress(epoch, total_epochs, batch, total_batches, checkpoint_index, total_loss, elapsed_s):
    return (
        f"[epoch {epoch}/{total_epochs}]"
        f"[batch {batch}/{total_batches}]"
        f"[ckpt {checkpoint_index}] "
        f"total_loss:{scalar_text(total_loss)}, elapsed:{elapsed_s}s"
    )


# CCFCRec
class CCFCRec(nn.Module):
    def __init__(self, args):
        super(CCFCRec, self).__init__()
        self.args = args
        self.method_variant = getattr(args, "method_variant", "baseline")
        self.category_conf_dim = int(getattr(args, "category_conf_dim", 16))
        self.category_conf_max_count = int(getattr(args, "category_conf_max_count", 5))
        self.category_gate_scale = float(getattr(args, "category_gate_scale", 0.5))
        self.m11r2_feature_dim = int(getattr(args, "m11r2_feature_dim", 16))
        self.m11r3_residual_max_ratio = float(getattr(args, "m11r3_residual_max_ratio", 0.15))
        self.m11r3_film_strength = float(getattr(args, "m11r3_film_strength", 0.1))
        self.m11r4_expert_film_strength = float(getattr(args, "m11r4_expert_film_strength", 0.2))
        self.m11r4_fusion_strength = float(getattr(args, "m11r4_fusion_strength", 0.25))
        self.cicp_feature_dim = int(getattr(args, "cicp_feature_dim", 16))
        self.cicp_residual_max_ratio = float(getattr(args, "cicp_residual_max_ratio", 0.15))
        self.cicp_modality_strength = float(getattr(args, "cicp_modality_strength", 0.25))
        self.cicp_expert_strength = float(getattr(args, "cicp_expert_strength", 0.20))
        self.cicp_attention_strength = float(getattr(args, "cicp_attention_strength", 0.50))
        self.cicpr2_residual_max_ratio = float(
            getattr(args, "cicpr2_residual_max_ratio", 0.15)
        )
        self.cicpr2_increment_strength = float(
            getattr(args, "cicpr2_increment_strength", 0.50)
        )
        self.cicpr2_cross_attention_strength = float(
            getattr(args, "cicpr2_cross_attention_strength", 0.50)
        )
        self.cicpr2_cross_attention_temperature = float(
            getattr(args, "cicpr2_cross_attention_temperature", 0.25)
        )
        self.cicpr2_category_dropout_max = float(
            getattr(args, "cicpr2_category_dropout_max", 0.50)
        )
        self._last_m11_residual = None
        self._last_cicp_residual = None
        self.category_bin_count = 4
        if self.uses_category_confidence():
            if self.category_conf_dim <= 0:
                raise ValueError("category_conf_dim must be positive for category confidence variants")
            if self.category_conf_max_count <= 0:
                raise ValueError("category_conf_max_count must be positive for category confidence variants")
        if self.uses_category_fusion_gate():
            if self.category_gate_scale <= 0 or self.category_gate_scale > 1:
                raise ValueError("category_gate_scale must be in (0, 1] for category_conf_fusion_gate")
        self.attr_matrix = torch.nn.Parameter(torch.FloatTensor(args.attr_num, args.attr_present_dim))
        # 定义属性attribute注意力层
        self.attr_W1 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, args.attr_present_dim))
        self.attr_b1 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, 1))
        self.attr_W2 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, 1))
        # 控制整个模型的激活函数
        self.h = nn.LeakyReLU()
        # 图像的映射矩阵
        self.image_projection = torch.nn.Parameter(torch.FloatTensor(4096, args.implicit_dim))
        self.sigmoid = torch.nn.Sigmoid()  # 将门控信号映射到[0, 1]之间
        # user和item的嵌入层，可用预训练的进行初始化
        if args.pretrain is True:
            if args.pretrain_update is True:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=True)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=True)
            else:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=False)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=False)
        else:
            self.user_embedding = nn.Parameter(torch.FloatTensor(args.user_number, args.implicit_dim))
            self.item_embedding = nn.Parameter(torch.FloatTensor(args.item_number, args.implicit_dim))
        # 定义生成层，将(q_v_a, u)的信息，共同生成 q_v_c， 生成包含协同信息的item嵌入
        if self.uses_category_confidence():
            self.category_conf_embedding = nn.Embedding(self.category_bin_count, self.category_conf_dim)
        if self.uses_category_fusion_gate():
            self.category_fusion_gate = nn.Linear(self.category_conf_extra_dim(), 1)
        if self.uses_m11_residual():
            self.m11r2_feature_projection = nn.Linear(M11_FEATURE_WIDTH, self.m11r2_feature_dim, bias=False)
            self.m11r2_feature_to_hidden = nn.Linear(
                self.m11r2_feature_dim,
                args.cat_implicit_dim,
                bias=False,
            )
        if self.uses_m11r3_dual_residual():
            self.m11r3_global_projection = nn.Linear(M11_FEATURE_WIDTH - 1, self.m11r2_feature_dim, bias=False)
            self.m11r3_global_to_hidden = nn.Linear(
                self.m11r2_feature_dim,
                args.cat_implicit_dim,
                bias=False,
            )
        if self.uses_m11r3_film():
            self.m11r3_film_projection = nn.Linear(M11_FEATURE_WIDTH, self.m11r2_feature_dim, bias=False)
            self.m11r3_film_scale = nn.Linear(self.m11r2_feature_dim, args.cat_implicit_dim, bias=False)
            self.m11r3_film_shift = nn.Linear(self.m11r2_feature_dim, args.cat_implicit_dim, bias=False)
        if self.uses_m11r4_protected_experts():
            self.m11r4_non_target_projection = nn.Linear(
                M11_FEATURE_WIDTH - 1,
                self.m11r2_feature_dim,
                bias=False,
            )
            self.m11r4_non_target_scale = nn.Linear(self.m11r2_feature_dim, args.cat_implicit_dim, bias=False)
            self.m11r4_non_target_shift = nn.Linear(self.m11r2_feature_dim, args.cat_implicit_dim, bias=False)
        if self.uses_m11r4_continuous_fusion():
            self.m11r4_fusion_projection = nn.Linear(M11_FEATURE_WIDTH, self.m11r2_feature_dim, bias=False)
            self.m11r4_attr_scale = nn.Linear(self.m11r2_feature_dim, args.attr_present_dim, bias=False)
            self.m11r4_image_scale = nn.Linear(self.m11r2_feature_dim, args.implicit_dim, bias=False)
        if self.uses_cicpr1_e4_residual():
            self.cicp_feature_projection = nn.Linear(
                CICP_FEATURE_WIDTH,
                self.cicp_feature_dim,
                bias=False,
            )
            self.cicp_feature_to_hidden = nn.Linear(
                self.cicp_feature_dim,
                args.cat_implicit_dim,
                bias=False,
            )
        if self.uses_cicpr1_modality_routing():
            self.cicp_modality_gate = nn.Linear(CICP_FEATURE_WIDTH, 1)
        if self.uses_cicpr1_category_expert():
            self.cicp_category_expert = nn.Linear(
                args.attr_present_dim,
                args.cat_implicit_dim,
                bias=False,
            )
            self.cicp_category_expert_gate = nn.Linear(CICP_FEATURE_WIDTH, 1)
        if self.uses_cicpr2_content_direction_residual():
            self.cicpr2_residual_category_direction = nn.Linear(
                args.attr_present_dim,
                args.cat_implicit_dim,
                bias=False,
            )
            self.cicpr2_residual_image_direction = nn.Linear(
                args.implicit_dim,
                args.cat_implicit_dim,
                bias=False,
            )
        if self.uses_cicpr2_cross_modal_attention():
            self.cicpr2_attention_image_query = nn.Linear(
                args.implicit_dim,
                args.attr_present_dim,
                bias=False,
            )
            self.cicpr2_attention_category_key = nn.Linear(
                args.attr_present_dim,
                args.attr_present_dim,
                bias=False,
            )
        if self.uses_cicpr2_score_distillation():
            score_hidden_dim = max(8, args.cat_implicit_dim // 4)
            self.cicpr2_score_head = nn.Sequential(
                nn.Linear(args.cat_implicit_dim, score_hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(score_hidden_dim, 1),
            )
        gen_input_dim = args.attr_present_dim + args.implicit_dim + self.category_conf_extra_dim()
        self.gen_layer1 = nn.Linear(gen_input_dim, args.cat_implicit_dim)
        self.gen_layer2 = nn.Linear(args.attr_present_dim, args.attr_present_dim)
        # 参数初始化
        self.__init_param__()

    def uses_category_confidence(self):
        return self.method_variant in {"category_conf_input", "category_conf_fusion_gate"}

    def uses_category_fusion_gate(self):
        return self.method_variant == "category_conf_fusion_gate"

    def uses_m11r2_feature_fusion(self):
        return self.method_variant in M11R2_FEATURE_METHOD_VARIANTS

    def uses_m11_residual(self):
        return self.method_variant in (
            {"m11r2_target_feature_fusion"}
            | M11R3_DUAL_RESIDUAL_METHOD_VARIANTS
            | M11R3_NORM_CAPPED_METHOD_VARIANTS
            | M11R3_NEIGHBOR_TRANSFER_METHOD_VARIANTS
            | M11R4_PROTECTED_EXPERT_METHOD_VARIANTS
        )

    def uses_m11r3_dual_residual(self):
        return self.method_variant in M11R3_DUAL_RESIDUAL_METHOD_VARIANTS

    def uses_m11r3_norm_cap(self):
        return self.method_variant in M11R3_NORM_CAPPED_METHOD_VARIANTS

    def uses_m11r3_neighbor_transfer(self):
        return self.method_variant in M11R3_NEIGHBOR_TRANSFER_METHOD_VARIANTS

    def uses_m11r3_film(self):
        return self.method_variant in M11R3_FILM_METHOD_VARIANTS

    def uses_m11r4_protected_experts(self):
        return self.method_variant in M11R4_PROTECTED_EXPERT_METHOD_VARIANTS

    def uses_m11r4_continuous_fusion(self):
        return self.method_variant in M11R4_CONTINUOUS_FUSION_METHOD_VARIANTS

    def uses_cicp_features(self):
        return self.method_variant in CICP_METHOD_VARIANTS

    def uses_cicpr1_e4_residual(self):
        return self.method_variant in CICPR1_E4_RESIDUAL_METHOD_VARIANTS

    def uses_cicpr1_modality_routing(self):
        return self.method_variant in CICPR1_MODALITY_ROUTING_METHOD_VARIANTS

    def uses_cicpr1_category_expert(self):
        return self.method_variant in CICPR1_CATEGORY_EXPERT_METHOD_VARIANTS

    def uses_cicpr1_adaptive_attention(self):
        return self.method_variant in CICPR1_ADAPTIVE_ATTENTION_METHOD_VARIANTS

    def uses_cicpr2_content_direction_residual(self):
        return self.method_variant in CICPR2_CONTENT_DIRECTION_RESIDUAL_METHOD_VARIANTS

    def uses_cicpr2_category_increment(self):
        return self.method_variant in CICPR2_CATEGORY_INCREMENT_METHOD_VARIANTS

    def uses_cicpr2_cross_modal_attention(self):
        return self.method_variant in CICPR2_CROSS_MODAL_ATTENTION_METHOD_VARIANTS

    def uses_cicpr2_score_distillation(self):
        return self.method_variant in CICPR2_SCORE_DISTILLATION_METHOD_VARIANTS

    def uses_cicpr2_reliability_dropout(self):
        return self.method_variant in CICPR2_RELIABILITY_DROPOUT_METHOD_VARIANTS

    def category_conf_extra_dim(self):
        if self.uses_category_confidence():
            return self.category_conf_dim + 2
        return 0

    def __init_param__(self):
        nn.init.xavier_normal_(self.attr_matrix)
        nn.init.xavier_normal_(self.attr_W1)
        nn.init.xavier_normal_(self.attr_W2)
        nn.init.xavier_normal_(self.attr_b1)
        nn.init.xavier_normal_(self.image_projection)
        # 生成层初始化
        # user, item嵌入层的初始化, 没有预训练的情况下就初始化
        if self.args.pretrain is False:
            nn.init.xavier_normal_(self.user_embedding)
            nn.init.xavier_normal_(self.item_embedding)
        nn.init.xavier_normal_(self.gen_layer1.weight)
        nn.init.xavier_normal_(self.gen_layer2.weight)
        if self.uses_category_confidence():
            nn.init.xavier_normal_(self.category_conf_embedding.weight)
        if self.uses_category_fusion_gate():
            nn.init.zeros_(self.category_fusion_gate.weight)
            nn.init.zeros_(self.category_fusion_gate.bias)
        if self.uses_m11_residual():
            nn.init.xavier_normal_(self.m11r2_feature_projection.weight)
            if self.uses_m11r3_neighbor_transfer():
                nn.init.zeros_(self.m11r2_feature_to_hidden.weight)
            else:
                nn.init.xavier_normal_(self.m11r2_feature_to_hidden.weight)
        if self.uses_m11r3_dual_residual():
            nn.init.xavier_normal_(self.m11r3_global_projection.weight)
            nn.init.zeros_(self.m11r3_global_to_hidden.weight)
        if self.uses_m11r3_film():
            nn.init.xavier_normal_(self.m11r3_film_projection.weight)
            nn.init.zeros_(self.m11r3_film_scale.weight)
            nn.init.zeros_(self.m11r3_film_shift.weight)
        if self.uses_m11r4_protected_experts():
            nn.init.xavier_normal_(self.m11r4_non_target_projection.weight)
            nn.init.zeros_(self.m11r4_non_target_scale.weight)
            nn.init.zeros_(self.m11r4_non_target_shift.weight)
        if self.uses_m11r4_continuous_fusion():
            nn.init.xavier_normal_(self.m11r4_fusion_projection.weight)
            nn.init.zeros_(self.m11r4_attr_scale.weight)
            nn.init.zeros_(self.m11r4_image_scale.weight)
        if self.uses_cicpr1_e4_residual():
            nn.init.xavier_normal_(self.cicp_feature_projection.weight)
            nn.init.xavier_normal_(self.cicp_feature_to_hidden.weight)
        if self.uses_cicpr1_modality_routing():
            nn.init.zeros_(self.cicp_modality_gate.weight)
            nn.init.zeros_(self.cicp_modality_gate.bias)
        if self.uses_cicpr1_category_expert():
            nn.init.xavier_normal_(self.cicp_category_expert.weight)
            nn.init.zeros_(self.cicp_category_expert_gate.weight)
            nn.init.constant_(self.cicp_category_expert_gate.bias, -2.0)
        if self.uses_cicpr2_content_direction_residual():
            nn.init.zeros_(self.cicpr2_residual_category_direction.weight)
            nn.init.zeros_(self.cicpr2_residual_image_direction.weight)
        if self.uses_cicpr2_cross_modal_attention():
            nn.init.xavier_normal_(self.cicpr2_attention_image_query.weight)
            nn.init.xavier_normal_(self.cicpr2_attention_category_key.weight)
        if self.uses_cicpr2_score_distillation():
            nn.init.xavier_normal_(self.cicpr2_score_head[0].weight)
            nn.init.zeros_(self.cicpr2_score_head[0].bias)
            nn.init.xavier_normal_(self.cicpr2_score_head[2].weight)
            nn.init.zeros_(self.cicpr2_score_head[2].bias)

    def build_category_conf_bins(self, attribute):
        category_count = (attribute != -1).sum(dim=1)
        bins = torch.zeros_like(category_count, dtype=torch.long)
        bins = torch.where((category_count > 0) & (category_count <= 3), torch.ones_like(bins), bins)
        bins = torch.where(category_count == 4, torch.full_like(bins, 2), bins)
        bins = torch.where(category_count >= 5, torch.full_like(bins, 3), bins)
        return bins

    def build_category_conf_features(self, attribute):
        category_count = (attribute != -1).sum(dim=1).float()
        count_clamped = torch.clamp(category_count, min=0, max=self.category_conf_max_count)
        category_density = count_clamped / float(self.category_conf_max_count)
        category_log_norm = torch.log1p(count_clamped) / math.log1p(float(self.category_conf_max_count))
        scalar_features = torch.stack((category_log_norm, category_density), dim=1)
        category_bin = self.build_category_conf_bins(attribute)
        category_conf_emb = self.category_conf_embedding(category_bin)
        return torch.cat((category_conf_emb, scalar_features.to(category_conf_emb.dtype)), dim=1)

    def apply_category_fusion_gate(self, final_attr_emb, p_v, category_conf_features):
        if not self.uses_category_fusion_gate():
            return final_attr_emb, p_v
        gate = torch.sigmoid(self.category_fusion_gate(category_conf_features))
        centered_gate = (2.0 * gate) - 1.0
        attr_scale = 1.0 + self.category_gate_scale * centered_gate
        image_scale = 1.0 - self.category_gate_scale * centered_gate
        return final_attr_emb * attr_scale, p_v * image_scale

    def build_generator_input(self, final_attr_emb, p_v, attribute):
        parts = [final_attr_emb, p_v]
        if self.uses_category_confidence():
            category_conf_features = self.build_category_conf_features(attribute)
            final_attr_emb, p_v = self.apply_category_fusion_gate(final_attr_emb, p_v, category_conf_features)
            parts = [final_attr_emb, p_v, category_conf_features]
        return torch.cat(parts, dim=1)

    def encode_content_components(
        self,
        attribute,
        image_feature,
        batch_size,
        m11_features=None,
        cicp_features=None,
    ):
        self._last_m11_residual = None
        self._last_cicp_residual = None
        features = None
        if self.uses_m11r2_feature_fusion():
            if m11_features is None:
                raise ValueError(f"m11_features are required for {self.method_variant}")
            if m11_features.ndim != 2 or m11_features.shape[1] != M11_FEATURE_WIDTH:
                raise ValueError(
                    f"m11_features must have shape [batch,{M11_FEATURE_WIDTH}], got {tuple(m11_features.shape)}"
                )
            features = m11_features.to(dtype=self.attr_matrix.dtype)
        cicp = None
        if self.uses_cicp_features():
            if cicp_features is None:
                raise ValueError(f"cicp_features are required for {self.method_variant}")
            if cicp_features.ndim != 2 or cicp_features.shape[1] != CICP_FEATURE_WIDTH:
                raise ValueError(
                    f"cicp_features must have shape [batch,{CICP_FEATURE_WIDTH}], "
                    f"got {tuple(cicp_features.shape)}"
                )
            cicp = cicp_features.to(dtype=self.attr_matrix.dtype)
        z_v = torch.matmul(torch.matmul(self.attr_matrix, self.attr_W1)+self.attr_b1.squeeze(), self.attr_W2)
        z_v_copy = z_v.repeat(batch_size, 1, 1)
        z_v_squeeze = z_v_copy.squeeze(dim=2)
        neg_inf = torch.full_like(z_v_squeeze, -1e6)
        z_v_mask = torch.where(attribute != -1, z_v_squeeze, neg_inf)
        if self.uses_cicpr1_adaptive_attention():
            centered_score = 2.0 * cicp[:, :1].clamp(0.0, 1.0) - 1.0
            attention_temperature = torch.exp(-self.cicp_attention_strength * centered_score)
            z_v_mask = z_v_mask / attention_temperature
        attr_attention_weight = torch.softmax(z_v_mask, dim=1)
        final_attr_emb = torch.matmul(attr_attention_weight, self.attr_matrix)
        p_v = torch.matmul(image_feature, self.image_projection)  # item的图像嵌入向量
        if self.uses_cicpr2_cross_modal_attention():
            image_query = F.normalize(
                self.cicpr2_attention_image_query(p_v),
                dim=1,
                eps=1e-6,
            )
            category_key = F.normalize(
                self.cicpr2_attention_category_key(self.attr_matrix),
                dim=1,
                eps=1e-6,
            )
            content_logits = torch.matmul(image_query, category_key.t()) / (
                self.cicpr2_cross_attention_temperature
            )
            content_logits = torch.where(attribute != -1, content_logits, neg_inf)
            content_attention = torch.softmax(content_logits, dim=1)
            content_attr_emb = torch.matmul(content_attention, self.attr_matrix)
            score_gate = (
                self.cicpr2_cross_attention_strength
                * cicp[:, :1].clamp(0.0, 1.0)
            )
            final_attr_emb = (
                (1.0 - score_gate) * final_attr_emb
                + score_gate * content_attr_emb
            )
        if self.uses_cicpr2_reliability_dropout() and self.training:
            score = cicp[:, :1].clamp(0.0, 1.0)
            keep_probability = 1.0 - self.cicpr2_category_dropout_max * (1.0 - score)
            keep_mask = torch.bernoulli(keep_probability)
            final_attr_emb = final_attr_emb * keep_mask / keep_probability.clamp_min(1e-6)
        if self.uses_cicpr1_modality_routing():
            delta = self.cicp_modality_strength * torch.tanh(self.cicp_modality_gate(cicp))
            final_attr_emb = final_attr_emb * (1.0 + delta)
            p_v = p_v * (1.0 - delta)
        if self.uses_m11r4_continuous_fusion():
            condition = self.h(self.m11r4_fusion_projection(features))
            signal = 0.5 + 0.5 * features[:, 1:2].clamp(0.0, 1.0)
            attr_scale = torch.tanh(self.m11r4_attr_scale(condition)) * signal
            image_scale = torch.tanh(self.m11r4_image_scale(condition)) * signal
            final_attr_emb = final_attr_emb * (1.0 + self.m11r4_fusion_strength * attr_scale)
            p_v = p_v * (1.0 + self.m11r4_fusion_strength * image_scale)
        q_v_a = self.build_generator_input(final_attr_emb, p_v, attribute)
        hidden = self.gen_layer1(q_v_a)
        if self.uses_cicpr2_category_increment():
            image_only_input = torch.cat((torch.zeros_like(final_attr_emb), p_v), dim=1)
            image_hidden = self.gen_layer1(image_only_input)
            category_increment = hidden - image_hidden
            score_gate = 1.0 + self.cicpr2_increment_strength * (
                2.0 * cicp[:, :1].clamp(0.0, 1.0) - 1.0
            )
            hidden = image_hidden + score_gate * category_increment
        if self.uses_cicpr1_e4_residual():
            cicp_residual = self.cicp_feature_to_hidden(
                self.h(self.cicp_feature_projection(cicp))
            )
            cicp_residual = cap_cicp_residual_norm(
                cicp_residual,
                hidden,
                self.cicp_residual_max_ratio,
            )
            self._last_cicp_residual = cicp_residual
            hidden = hidden + cicp_residual
        elif self.uses_cicpr2_content_direction_residual():
            content_direction = torch.tanh(
                self.cicpr2_residual_category_direction(final_attr_emb)
                + self.cicpr2_residual_image_direction(p_v)
            )
            cicp_residual = cicp[:, :1].clamp(0.0, 1.0) * content_direction
            cicp_residual = cap_cicp_residual_norm(
                cicp_residual,
                hidden,
                self.cicpr2_residual_max_ratio,
            )
            self._last_cicp_residual = cicp_residual
            hidden = hidden + cicp_residual
        elif self.uses_cicpr1_category_expert():
            expert_hidden = self.cicp_category_expert(final_attr_emb)
            expert_gate = (
                self.cicp_expert_strength
                * cicp[:, :1].clamp(0.0, 1.0)
                * torch.sigmoid(self.cicp_category_expert_gate(cicp))
            )
            hidden = (1.0 - expert_gate) * hidden + expert_gate * expert_hidden
        if self.uses_m11r2_feature_fusion():
            features = features.to(hidden.dtype)
            if self.uses_m11r3_film():
                condition = self.h(self.m11r3_film_projection(features))
                scale = torch.tanh(self.m11r3_film_scale(condition))
                shift = torch.tanh(self.m11r3_film_shift(condition))
                hidden_rms = hidden.detach().pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
                modulated = (
                    hidden * (1.0 + self.m11r3_film_strength * scale)
                    + self.m11r3_film_strength * hidden_rms * shift
                )
                self._last_m11_residual = modulated - hidden
                hidden = modulated
            elif self.uses_m11r4_protected_experts():
                base_hidden = hidden
                target_gate = features[:, :1].clamp(0.0, 1.0)
                non_target_gate = 1.0 - target_gate
                target_residual = self.m11r2_feature_to_hidden(
                    self.h(self.m11r2_feature_projection(features * target_gate))
                ) * target_gate
                target_hidden = base_hidden + target_residual

                non_target_condition = self.h(self.m11r4_non_target_projection(features[:, 1:]))
                non_target_scale = torch.tanh(self.m11r4_non_target_scale(non_target_condition))
                non_target_shift = torch.tanh(self.m11r4_non_target_shift(non_target_condition))
                hidden_rms = base_hidden.detach().pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
                non_target_hidden = (
                    base_hidden * (1.0 + self.m11r4_expert_film_strength * non_target_scale)
                    + self.m11r4_expert_film_strength * hidden_rms * non_target_shift
                )
                hidden = target_gate * target_hidden + non_target_gate * non_target_hidden
                self._last_m11_residual = hidden - base_hidden
            elif self.uses_m11_residual():
                residual_features = features
                if self.uses_m11r3_dual_residual():
                    target_gate = features[:, :1]
                    residual_features = features * target_gate
                feature_hidden = self.m11r2_feature_to_hidden(
                    self.h(self.m11r2_feature_projection(residual_features))
                )
                if self.uses_m11r3_dual_residual():
                    global_hidden = self.m11r3_global_to_hidden(
                        self.h(self.m11r3_global_projection(features[:, 1:]))
                    )
                    feature_hidden = feature_hidden + global_hidden
                if self.uses_m11r3_norm_cap():
                    feature_hidden = cap_m11_residual_norm(
                        feature_hidden,
                        hidden,
                        self.m11r3_residual_max_ratio,
                    )
                self._last_m11_residual = feature_hidden
                hidden = hidden + feature_hidden
        q_v_c = self.gen_layer2(self.h(hidden))
        return q_v_c, final_attr_emb, p_v

    def predict_cicpr2_score(self, generated_embedding):
        if not self.uses_cicpr2_score_distillation():
            raise ValueError("CICP-R2 score head is only available for score distillation")
        return torch.sigmoid(self.cicpr2_score_head(generated_embedding)).squeeze(dim=1)

    def forward(self, attribute, image_feature, batch_size, m11_features=None, cicp_features=None):
        q_v_c, _, _ = self.encode_content_components(
            attribute,
            image_feature,
            batch_size,
            m11_features=m11_features,
            cicp_features=cicp_features,
        )
        return q_v_c


def train(model, train_loader, optimizer, valida, args, model_save_dir):
    print("model start train!")
    test_save_path = model_save_dir + "/result.csv"
    print("model train at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # 写入超参数
    with open(model_save_dir + "/readme.txt", 'a+') as f:
        str_ = args_tostring(args)
        f.write(str_)
        f.write('\nsave dir:'+model_save_dir)
        f.write('\nmodel train time:'+(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    write_run_config(args, model, model_save_dir)
    with open(test_save_path, 'a+') as f:
        f.write(training_result_header())
    save_index = 0
    total_batches = len(train_loader)
    model = model.to(device)
    non_blocking = device.type == 'cuda' and getattr(args, 'pin_memory', False)
    item_category_tensor = train_loader.dataset.item_category_tensor.to(device, non_blocking=non_blocking)
    item_image_feature_tensor = train_loader.dataset.item_image_feature_tensor.to(device, non_blocking=non_blocking)
    task4_item_weights = load_task4_item_weights(
        train_loader.dataset.item_serialize_dict,
        args,
        item_number=train_loader.dataset.item_number,
    )
    if task4_item_weights is not None:
        task4_item_weights = task4_item_weights.to(device, non_blocking=non_blocking)
    m11r2_target_scores = load_m11r2_target_score_tensor(
        train_loader.dataset.item_serialize_dict,
        args,
        item_number=train_loader.dataset.item_number,
    )
    if m11r2_target_scores is not None:
        m11r2_target_scores = m11r2_target_scores.to(device, non_blocking=non_blocking)
    m11r2_feature_tensor = load_m11r2_feature_tensor(
        train_loader.dataset.item_serialize_dict,
        args,
        item_number=train_loader.dataset.item_number,
    )
    if m11r2_feature_tensor is not None:
        m11r2_feature_tensor = m11r2_feature_tensor.to(device, non_blocking=non_blocking)
    cicpr1_feature_tensor = load_cicpr1_feature_tensor(
        train_loader.dataset.item_serialize_dict,
        args,
        item_number=train_loader.dataset.item_number,
    )
    if cicpr1_feature_tensor is not None:
        cicpr1_feature_tensor = cicpr1_feature_tensor.to(device, non_blocking=non_blocking)
    task4_pair_margin_targets = load_task4_pair_margin_targets(
        train_loader.dataset.item_serialize_dict,
        args,
        item_number=train_loader.dataset.item_number,
    )
    if task4_pair_margin_targets is not None:
        task4_pair_margin_targets = {
            name: tensor.to(device, non_blocking=non_blocking)
            for name, tensor in task4_pair_margin_targets.items()
        }
    task4_competitor_pair_targets = load_task4_competitor_pair_targets(
        train_loader.dataset.item_serialize_dict,
        args,
        item_number=train_loader.dataset.item_number,
    )
    if task4_competitor_pair_targets is not None:
        task4_competitor_pair_targets = {
            name: tensor.to(device, non_blocking=non_blocking)
            for name, tensor in task4_competitor_pair_targets.items()
        }
    task4_boundary_competitors = load_task4_boundary_competitors(
        train_loader.dataset.item_serialize_dict,
        train_loader.dataset.user_serialize_dict,
        args,
        item_number=train_loader.dataset.item_number,
    )
    if task4_boundary_competitors is not None:
        task4_boundary_competitors = task4_boundary_competitors.to(device, non_blocking=non_blocking)
    for i_epoch in range(args.epoch):
        i_batch = 0
        batch_time = time.time()
        for user, item, neg_user, positive_item_list, negative_item_list, self_neg_list, support_confidence in tqdm(train_loader):
            optimizer.zero_grad()
            model.train()
            # allocate memory cpu to gpu
            user = user.to(device, non_blocking=non_blocking)
            item = item.to(device, non_blocking=non_blocking)
            neg_user = neg_user.to(device, non_blocking=non_blocking)
            positive_item_list = positive_item_list.to(device, non_blocking=non_blocking)
            negative_item_list = negative_item_list.to(device, non_blocking=non_blocking)
            self_neg_list = self_neg_list.to(device, non_blocking=non_blocking)
            support_confidence = support_confidence.to(device, non_blocking=non_blocking)
            item_genres = item_category_tensor[item]
            item_img_feature = item_image_feature_tensor[item]
            task4_batch_weights = task4_item_weights[item] if task4_item_weights is not None else None
            if getattr(args, "method_variant", "baseline") == "m11r2_qbpr_curriculum":
                task4_batch_weights = build_m11r2_curriculum_weights(
                    task4_batch_weights,
                    i_epoch,
                    getattr(args, "m11r2_curriculum_warmup_epochs", 20),
                )
            m11r2_batch_target_scores = (
                m11r2_target_scores[item] if m11r2_target_scores is not None else None
            )
            m11r2_batch_features = (
                m11r2_feature_tensor[item] if m11r2_feature_tensor is not None else None
            )
            cicpr1_batch_features = (
                cicpr1_feature_tensor[item] if cicpr1_feature_tensor is not None else None
            )
            task4_batch_pair_margin_targets = (
                {name: tensor[item] for name, tensor in task4_pair_margin_targets.items()}
                if task4_pair_margin_targets is not None
                else None
            )
            task4_batch_competitor_pair_targets = (
                {name: tensor[item] for name, tensor in task4_competitor_pair_targets.items()}
                if task4_competitor_pair_targets is not None
                else None
            )
            # run model
            q_v_c = model(
                item_genres,
                item_img_feature,
                user.shape[0],
                m11_features=m11r2_batch_features,
                cicp_features=cicpr1_batch_features,
            )
            cicpr2_distillation_sum = q_v_c.new_tensor(0.0)
            if (
                getattr(args, "method_variant", "baseline")
                in CICPR2_SCORE_DISTILLATION_METHOD_VARIANTS
            ):
                predicted_cicp_score = model.predict_cicpr2_score(q_v_c)
                cicpr2_distillation_sum = build_cicpr2_score_distillation_loss(
                    predicted_cicp_score,
                    cicpr1_batch_features,
                ) * float(getattr(args, "cicpr2_distillation_weight", 0.05))
            cicpr2_ordinal_sum = q_v_c.new_tensor(0.0)
            if (
                getattr(args, "method_variant", "baseline")
                in CICPR2_ORDINAL_COUNTERFACTUAL_METHOD_VARIANTS
                and item_genres.shape[0] > 1
            ):
                shuffled_item_genres = torch.roll(item_genres, shifts=1, dims=0)
                shuffled_q_v_c = model(
                    shuffled_item_genres,
                    item_img_feature,
                    user.shape[0],
                    cicp_features=cicpr1_batch_features,
                )
                cicpr2_ordinal_sum = build_cicpr2_ordinal_counterfactual_loss(
                    q_v_c,
                    shuffled_q_v_c,
                    model.item_embedding[item],
                    cicpr1_batch_features,
                    args,
                ) * float(getattr(args, "cicpr2_ordinal_weight", 0.05))
            cicpr1_alignment_sum = q_v_c.new_tensor(0.0)
            if getattr(args, "method_variant", "baseline") in CICPR1_ALIGNMENT_METHOD_VARIANTS:
                cicpr1_alignment_sum = build_cicpr1_alignment_loss(
                    q_v_c,
                    model.item_embedding[item],
                    cicpr1_batch_features,
                    i_epoch,
                    args,
                ) * float(getattr(args, "cicp_alignment_weight", 0.05))
            m11r3_neighbor_transfer_sum = q_v_c.new_tensor(0.0)
            if getattr(args, "method_variant", "baseline") in M11R3_NEIGHBOR_TRANSFER_METHOD_VARIANTS:
                m11r3_neighbor_transfer_sum = build_m11r3_neighbor_transfer_loss(
                    model._last_m11_residual,
                    m11r2_batch_features,
                    temperature=getattr(args, "m11r3_neighbor_temperature", 0.25),
                ) * float(getattr(args, "m11r3_neighbor_loss_weight", 0.1))
            m11r4_relational_alignment_sum = q_v_c.new_tensor(0.0)
            if getattr(args, "method_variant", "baseline") in M11R4_RELATIONAL_ALIGNMENT_METHOD_VARIANTS:
                m11r4_relational_alignment_sum = build_m11r4_relational_alignment_loss(
                    q_v_c,
                    m11r2_batch_features,
                ) * float(getattr(args, "m11r4_relation_loss_weight", 0.05))
            q_v_c_unsqueeze = q_v_c.unsqueeze(dim=1)
            # compute contrast loss
            positive_item_emb = model.item_embedding[positive_item_list]
            pos_contrast_mul = torch.sum(torch.mul(q_v_c_unsqueeze, positive_item_emb), dim=2) / (
                    args.tau * torch.norm(q_v_c_unsqueeze, dim=2) * torch.norm(positive_item_emb, dim=2))
            pos_contrast_exp = torch.exp(pos_contrast_mul)  # shape = 1024*10
            # negative samples
            neg_item_emb = model.item_embedding[negative_item_list]
            q_v_c_un2squeeze = q_v_c_unsqueeze.unsqueeze(dim=1)
            neg_contrast_mul = torch.sum(torch.mul(q_v_c_un2squeeze, neg_item_emb), dim=3) / (
                    args.tau * torch.norm(q_v_c_un2squeeze, dim=3) * torch.norm(neg_item_emb, dim=3))
            neg_contrast_exp = torch.exp(neg_contrast_mul)
            neg_contrast_sum = torch.sum(neg_contrast_exp, dim=2)  # shape = [1024, 10]
            contrast_val = -torch.log(pos_contrast_exp / (pos_contrast_exp + neg_contrast_sum))  # shape = [1024*10]
            contrast_examples_num = contrast_val.shape[0] * contrast_val.shape[1]
            contrast_per_item = torch.sum(contrast_val, dim=1) / contrast_val.shape[1]
            category_weights = build_category_reweight(item_genres, args)
            m11r4_focal_temperature = float(getattr(args, "m11r4_focal_temperature", 0.5))
            m11r4_contrast_weights = None
            if getattr(args, "method_variant", "baseline") in M11R4_CONTINUOUS_FOCAL_METHOD_VARIANTS:
                contrast_difficulty = 1.0 - torch.exp(
                    -contrast_per_item.detach().clamp_min(0.0) / m11r4_focal_temperature
                )
                m11r4_contrast_weights = build_m11r4_continuous_focal_weights(
                    contrast_difficulty,
                    m11r2_batch_features,
                    args,
                )
            if m11r4_contrast_weights is not None:
                contrast_sum = weighted_sum(contrast_per_item, m11r4_contrast_weights, True)
            elif task4_batch_weights is not None:
                contrast_sum = weighted_sum(contrast_per_item, task4_batch_weights, getattr(args, "task4_reweight_contrast", False))
            else:
                contrast_sum = weighted_sum(contrast_per_item, category_weights, args.reweight_contrast)
            '''
            contrast self
            '''
            self_neg_item_emb = model.item_embedding[self_neg_list]
            self_neg_contrast_mul = torch.sum(torch.mul(q_v_c_unsqueeze, self_neg_item_emb), dim=2)/(
                args.tau*torch.norm(q_v_c_unsqueeze, dim=2)*torch.norm(self_neg_item_emb, dim=2))
            self_neg_contrast_sum = torch.sum(torch.exp(self_neg_contrast_mul), dim=1)
            item_emb = model.item_embedding[item]
            self_pos_contrast_mul = torch.sum(torch.mul(q_v_c, item_emb), dim=1) / (
                    args.tau * torch.norm(q_v_c, dim=1) * torch.norm(item_emb, dim=1))
            self_pos_contrast_exp = torch.exp(self_pos_contrast_mul)  # shape = 1024*1
            self_contrast_val = -torch.log(self_pos_contrast_exp/(self_pos_contrast_exp+self_neg_contrast_sum))
            m11r4_self_weights = None
            if getattr(args, "method_variant", "baseline") in M11R4_CONTINUOUS_FOCAL_METHOD_VARIANTS:
                self_difficulty = 1.0 - torch.exp(
                    -self_contrast_val.detach().clamp_min(0.0) / m11r4_focal_temperature
                )
                m11r4_self_weights = build_m11r4_continuous_focal_weights(
                    self_difficulty,
                    m11r2_batch_features,
                    args,
                )
            if m11r4_self_weights is not None:
                self_contrast_sum = weighted_sum(self_contrast_val, m11r4_self_weights, True)
            elif task4_batch_weights is not None:
                self_contrast_sum = weighted_sum(
                    self_contrast_val,
                    task4_batch_weights,
                    not getattr(args, "task4_disable_self_contrast_weight", False),
                )
            else:
                self_contrast_sum = weighted_sum(self_contrast_val, category_weights, args.reweight_self_contrast)
            # rank loss
            user_emb = model.user_embedding[user]
            item_emb = model.item_embedding[item]
            neg_user_emb = model.user_embedding[neg_user]
            logsigmoid = torch.nn.LogSigmoid()
            y_uv = torch.mul(item_emb, user_emb).sum(dim=1)
            y_kv = torch.mul(item_emb, neg_user_emb).sum(dim=1)
            y_ukv = -logsigmoid(y_uv - y_kv).sum()
            # 使用属性生成item嵌入，再做一个bpr排序
            y_uv2 = torch.mul(q_v_c, user_emb).sum(dim=1)
            y_kv2 = torch.mul(q_v_c, neg_user_emb).sum(dim=1)
            y_q_margin_diff = y_uv2 - y_kv2
            y_ukv2_per_item = -logsigmoid(y_uv2 - y_kv2)
            cicpr1_counterfactual_sum = q_v_c.new_tensor(0.0)
            if (
                getattr(args, "method_variant", "baseline")
                in CICPR1_COUNTERFACTUAL_METHOD_VARIANTS
                and item_genres.shape[0] > 1
            ):
                shuffled_item_genres = torch.roll(item_genres, shifts=1, dims=0)
                shuffled_q_v_c = model(
                    shuffled_item_genres,
                    item_img_feature,
                    user.shape[0],
                    cicp_features=cicpr1_batch_features,
                )
                shuffled_positive_score = torch.mul(shuffled_q_v_c, user_emb).sum(dim=1)
                shuffled_negative_score = torch.mul(shuffled_q_v_c, neg_user_emb).sum(dim=1)
                shuffled_margin = shuffled_positive_score - shuffled_negative_score
                cicpr1_counterfactual_sum = build_cicpr1_counterfactual_margin_loss(
                    y_q_margin_diff,
                    shuffled_margin,
                    cicpr1_batch_features,
                    args,
                ) * float(getattr(args, "cicp_counterfactual_weight", 0.05))
            adaptive_qbpr_weights = build_adaptive_qbpr_weights(item_genres, support_confidence, args)
            m11r4_qbpr_weights = None
            if getattr(args, "method_variant", "baseline") in M11R4_CONTINUOUS_FOCAL_METHOD_VARIANTS:
                qbpr_difficulty = torch.sigmoid(
                    -y_q_margin_diff.detach() / m11r4_focal_temperature
                )
                m11r4_qbpr_weights = build_m11r4_continuous_focal_weights(
                    qbpr_difficulty,
                    m11r2_batch_features,
                    args,
                )
            if m11r4_qbpr_weights is not None:
                y_ukv2 = weighted_sum(y_ukv2_per_item, m11r4_qbpr_weights, True)
            elif adaptive_qbpr_weights is not None:
                y_ukv2 = weighted_sum(y_ukv2_per_item, adaptive_qbpr_weights, True)
            elif m11r2_batch_target_scores is not None:
                focal_weights = build_m11r2_focal_qbpr_weights(
                    y_q_margin_diff,
                    m11r2_batch_target_scores,
                    args,
                )
                y_ukv2 = weighted_sum(y_ukv2_per_item, focal_weights, True)
            elif task4_batch_weights is not None:
                y_ukv2 = weighted_sum(
                    y_ukv2_per_item,
                    task4_batch_weights,
                    not getattr(args, "task4_disable_q_bpr_weight", False),
                )
            else:
                y_ukv2 = weighted_sum(y_ukv2_per_item, category_weights, args.reweight_q_bpr)
            task4_pair_margin_sum = task4_pair_margin_loss(y_q_margin_diff, task4_batch_pair_margin_targets)
            task4_competitor_user = resolve_task4_competitor_user_ids(item, neg_user, task4_boundary_competitors)
            task4_competitor_user_emb = model.user_embedding[task4_competitor_user]
            task4_competitor_score = torch.mul(q_v_c, task4_competitor_user_emb).sum(dim=1)
            task4_competitor_margin_diff = y_uv2 - task4_competitor_score
            task4_competitor_pair_sum = task4_competitor_pair_loss(
                task4_competitor_margin_diff,
                task4_batch_competitor_pair_targets,
            )
            total_loss = (
                args.lambda1*(contrast_sum+self_contrast_sum)
                + (1-args.lambda1)*(y_ukv+y_ukv2)
                + task4_pair_margin_sum
                + task4_competitor_pair_sum
                + m11r3_neighbor_transfer_sum
                + m11r4_relational_alignment_sum
                + cicpr1_alignment_sum
                + cicpr1_counterfactual_sum
                + cicpr2_distillation_sum
                + cicpr2_ordinal_sum
            )
            if math.isnan(total_loss):
                print("loss is nan!, exit.", total_loss)
                exit(255)
            total_loss.backward()
            optimizer.step()
            i_batch += 1
            if i_batch % args.save_batch_time == 0:
                model.eval()
                elapsed_s = int(time.time()-batch_time)
                checkpoint_index = save_index + 1
                print(format_training_progress(
                    i_epoch + 1,
                    args.epoch,
                    i_batch,
                    total_batches,
                    checkpoint_index,
                    total_loss,
                    elapsed_s,
                ))
                with torch.no_grad():
                    hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20 = valida.start_validate(model)
                with open(test_save_path, 'a+') as f:
                    f.write(build_training_result_row(
                        checkpoint_index,
                        i_epoch + 1,
                        i_batch,
                        total_batches,
                        elapsed_s,
                        total_loss,
                        contrast_sum,
                        (hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20),
                    ))
                # 保存模型
                batch_time = time.time()
                save_index = checkpoint_index
                torch.save(model.state_dict(), model_save_dir + '/' + str(save_index)+".pt")


if __name__ == '__main__':
    # args
    args = get_args()
    validate_method_args(args)
    # result save dir
    save_dir = os.path.join(args.result_root, time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time())))
    os.makedirs(save_dir, exist_ok=False)
    set_random_seed(args.seed)
    print("progress start at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    train_path = "data/train_withneg_rating.csv"
    vliad_path = 'data/validate_rating.csv'
    train_df = pd.read_csv(train_path)
    total_user_set = train_df['reviewerID']
    user_ser_dict = serialize_user(total_user_set)
    asin_category_int_map, category_ser_map = serial_asin_category()
    img_feature_dict = get_img_feature_pickle()
    # write internal variable
    with open(save_dir+"/save_dict.pkl", "wb") as file:
        save_dict = {'img_feature_dict': img_feature_dict, 'asin_category_int_map': asin_category_int_map,
                     'category_ser_map_len': category_ser_map.__len__(), 'user_ser_dict': user_ser_dict}
        pickle.dump(save_dict, file)
    # load dataset
    dataSet = RatingDataset(train_df, img_feature_dict, asin_category_int_map, category_ser_map.__len__(),
                            user_ser_dict, args.positive_number, args.negative_number,
                            adaptive_history_max_count=args.adaptive_history_max_count,
                            negative_sampling_mode=args.negative_sampling_mode,
                            negative_sampling_cache_size=args.negative_sampling_cache_size)
    args.user_number = dataSet.user_number
    args.item_number = dataSet.item_number
    loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
    }
    if args.pin_memory:
        loader_kwargs['pin_memory'] = True
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = args.persistent_workers
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
        if args.multiprocessing_context:
            loader_kwargs['multiprocessing_context'] = args.multiprocessing_context
    if args.seed >= 0:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        loader_kwargs['generator'] = generator
        loader_kwargs['worker_init_fn'] = functools.partial(seed_worker, base_seed=args.seed)
    train_loader = torch.utils.data.DataLoader(dataSet, **loader_kwargs)
    print("模型超参数:", args_tostring(args))
    myModel = CCFCRec(args)
    optimizer = torch.optim.Adam(myModel.parameters(), lr=args.learning_rate, weight_decay=0.1)
    validator = Validate(
        validate_csv=vliad_path,
        user_serialize_dict=user_ser_dict,
        img=img_feature_dict,
        genres=asin_category_int_map,
        category_num=category_ser_map.__len__(),
        batch_size=args.validate_batch_size,
        task4_profile_path=args.task4_profile_path,
        use_m11_features=uses_m11r2_feature_fusion(args),
        m11_feature_mode=resolve_m11_feature_mode(args),
        reject_m11_evaluation_columns=(
            getattr(args, "method_variant", "baseline") in M11R4_FEATURE_METHOD_VARIANTS
        ),
        cicp_profile_path=args.cicp_profile_path,
        use_cicp_features=uses_cicp_features(args),
        reject_cicp_evaluation_columns=True,
    )
    train(myModel, train_loader, optimizer, validator, args, save_dir)

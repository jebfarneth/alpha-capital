"""Train Stage-1 per-pattern ML rankers from frozen manifests.

The job is deliberately offline and shadow-first. It reads manifest-governed
signals, selects already-materialized stored features through
``alpha.ml.model_features``, trains one GBRT per pattern, writes a model
artifact, and registers that artifact for later shadow inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload

from alpha.db.engine import get_session
from alpha.db.models import (
    FeatureSnapshot,
    ForwardReturnObservation,
    MLModelRegistry,
    SignalRegistry,
)
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.runner import run_job
from alpha.market_calendar import is_us_equity_session, next_us_equity_session
from alpha.ml.cv import (
    CVExample,
    purged_embargoed_walk_forward_splits,
    unique_name_weights,
)
from alpha.ml.manifest_loader import FrozenMLManifest, PatternManifest, load_manifest
from alpha.ml.model_features import (
    SelectedFeatureVector,
    audit_feature_schema_no_leakage,
    feature_schema_hash,
    select_features,
)
from alpha.security_identity import (
    ResolvedSecurityIdentity,
    resolve_security_identities_for_tickers,
)


MODEL_FAMILY = "sklearn_hist_gradient_boosting_gbrt"
DEFAULT_STATUS = "shadow"
REJECTED_STATUS = "rejected"
LIFT_POPULATION_MEAN_FLOOR = 1e-6
PREDICTION_VARIANCE_EPSILON = 1e-12
WEIGHT_BOUNDARY_EPSILON = 1e-12
DEFAULT_MIN_SAMPLES_LEAF = 20
DEFAULT_L2_REGULARIZATION = 0.01
DEFAULT_EARLY_STOPPING = True
DEFAULT_DROP_FRACTION_FLAG_THRESHOLD = 0.20
DEFAULT_DROP_LABEL_MEAN_DELTA_THRESHOLD = 0.05
DEFAULT_MIN_CV_TRAIN_ROWS = 2
DEFAULT_MIN_CV_TRAIN_SECURITIES = 2
DEFAULT_EARLY_STOPPING_MIN_TRAIN_ROWS = 20
DEFAULT_TRAINER_DB_TIMEOUT_MS = 3_600_000
FEATURE_SNAPSHOT_PRELOAD_CHUNK_SIZE = 500
DEFAULT_OOS_GATE = {
    "min_top_decile_lift": 1.0,
    "min_rank_ic": 0.0,
    "required_metrics": ["top_decile_lift", "rank_ic"],
    "reject_status": REJECTED_STATUS,
}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingExample(CVExample):
    label: float
    vector: SelectedFeatureVector
    direction: str = "long"
    raw_label: float = math.nan
    realized_label_window_sessions: int | None = None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(str(value)[:10])


def _signal_date(signal: SignalRegistry) -> date:
    parsed = _parse_date(signal.trading_date)
    if parsed is not None:
        return parsed
    return signal.signal_timestamp.date()


def _finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _assert_finite_metric_values(name: str, values: list[float]) -> None:
    bad = [idx for idx, value in enumerate(values) if not math.isfinite(value)]
    if bad:
        raise ValueError(
            f"{name} contains non-finite values at indexes {bad[:5]}"
        )


def _assert_metric_inputs(
    y_true: list[float],
    y_pred: list[float],
    weights: list[float] | None = None,
) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(
            "metric inputs must have matching lengths: "
            f"y_true={len(y_true)} y_pred={len(y_pred)}"
        )
    _assert_finite_metric_values("y_true", y_true)
    _assert_finite_metric_values("y_pred", y_pred)
    if weights is None:
        return
    if len(weights) != len(y_true):
        raise ValueError(
            "metric weights must match label length: "
            f"weights={len(weights)} y_true={len(y_true)}"
        )
    _assert_finite_metric_values("weights", weights)
    bad_weights = [idx for idx, weight in enumerate(weights) if weight < 0.0]
    if bad_weights:
        raise ValueError(
            f"weights contains negative values at indexes {bad_weights[:5]}"
        )


def _feature_ranges(matrix: list[list[float]]) -> list[dict[str, float | None]]:
    if not matrix:
        return []
    out: list[dict[str, float | None]] = []
    for col_idx in range(len(matrix[0])):
        col = _finite([row[col_idx] for row in matrix])
        out.append(
            {
                "min": min(col) if col else None,
                "max": max(col) if col else None,
            }
        )
    return out


def _weighted_mae(y_true: list[float], y_pred: list[float], weights: list[float]) -> float:
    _assert_metric_inputs(y_true, y_pred, weights)
    denom = sum(weights)
    if denom <= 0:
        return math.nan
    return sum(abs(a - b) * w for a, b, w in zip(y_true, y_pred, weights)) / denom


def _pearson(y_true: list[float], y_pred: list[float]) -> float:
    _assert_metric_inputs(y_true, y_pred)
    if len(y_true) < 2:
        return math.nan
    if not _prediction_has_variance(y_true) or not _prediction_has_variance(y_pred):
        return math.nan
    mean_true = sum(y_true) / len(y_true)
    mean_pred = sum(y_pred) / len(y_pred)
    num = sum((a - mean_true) * (b - mean_pred) for a, b in zip(y_true, y_pred))
    den_a = math.sqrt(sum((a - mean_true) ** 2 for a in y_true))
    den_b = math.sqrt(sum((b - mean_pred) ** 2 for b in y_pred))
    if den_a == 0 or den_b == 0:
        return math.nan
    return num / (den_a * den_b)


def _ranks(values: list[float]) -> list[float]:
    _assert_finite_metric_values("values", values)
    out = [math.nan] * len(values)
    ordered = sorted(enumerate(values), key=lambda row: row[1])
    idx = 0
    while idx < len(ordered):
        end = idx + 1
        while (
            end < len(ordered)
            and abs(ordered[end][1] - ordered[end - 1][1])
            <= PREDICTION_VARIANCE_EPSILON
        ):
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for original_idx, _value in ordered[idx:end]:
            out[original_idx] = avg_rank
        idx = end
    return out


def _spearman(y_true: list[float], y_pred: list[float]) -> float:
    _assert_metric_inputs(y_true, y_pred)
    if len(y_true) < 2:
        return math.nan
    return _pearson(_ranks(y_true), _ranks(y_pred))


def _top_quantile_mean(
    y_true: list[float], y_pred: list[float], *, quantile: float = 0.2
) -> float:
    _assert_metric_inputs(y_true, y_pred)
    if not y_true:
        return math.nan
    if _top_quantile_unreliable(y_pred, quantile=quantile):
        return math.nan
    count = max(1, int(math.ceil(len(y_true) * quantile)))
    ranked = _ranked_prediction_rows(y_true, y_pred)
    return _mean([label for _pred, label, _weight in ranked[-count:]])


def _top_quantile_unreliable(y_pred: list[float], *, quantile: float = 0.2) -> bool:
    _assert_finite_metric_values("y_pred", y_pred)
    if not y_pred:
        return True
    count = max(1, int(math.ceil(len(y_pred) * quantile)))
    boundary = len(y_pred) - count
    return _row_count_boundary_unreliable(y_pred, boundary)


def _mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if len(values) != len(weights):
        raise ValueError(
            "weighted mean inputs must have matching lengths: "
            f"values={len(values)} weights={len(weights)}"
        )
    _assert_finite_metric_values("values", values)
    _assert_finite_metric_values("weights", weights)
    denom = sum(weights)
    if not values or denom <= 0:
        return math.nan
    return sum(value * weight for value, weight in zip(values, weights)) / denom


def _median(values: list[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _win_rate(values: list[float]) -> float:
    if not values:
        return math.nan
    return sum(1 for value in values if value > 0.0) / len(values)


def _weighted_win_rate(values: list[float], weights: list[float]) -> float:
    denom = sum(weights)
    if not values or denom <= 0:
        return math.nan
    return sum(weight for value, weight in zip(values, weights) if value > 0.0) / denom


def _ranked_prediction_rows(
    y_true: list[float],
    y_pred: list[float],
    weights: list[float] | None = None,
) -> list[tuple[float, float, float]]:
    _assert_metric_inputs(y_true, y_pred, weights)
    row_weights = weights or [1.0 for _ in y_true]
    return sorted(zip(y_pred, y_true, row_weights), key=lambda row: row[0])


def _score_percentiles(y_pred: list[float]) -> list[float]:
    _assert_finite_metric_values("y_pred", y_pred)
    if not y_pred:
        return []
    rows = sorted(enumerate(y_pred), key=lambda row: row[1])
    out = [math.nan] * len(y_pred)
    idx = 0
    while idx < len(rows):
        end = idx + 1
        while (
            end < len(rows)
            and abs(rows[end][1] - rows[end - 1][1])
            <= PREDICTION_VARIANCE_EPSILON
        ):
            end += 1
        percentile = (idx + end) / (2.0 * len(y_pred))
        for original_idx, _pred in rows[idx:end]:
            out[original_idx] = percentile
        idx = end
    return out


def _prediction_has_variance(y_pred: list[float]) -> bool:
    finite = [value for value in y_pred if math.isfinite(value)]
    if len(finite) < 2:
        return False
    return max(finite) - min(finite) > PREDICTION_VARIANCE_EPSILON


def _row_count_boundary_unreliable(
    y_pred: list[float],
    boundary: int,
) -> bool:
    if not _prediction_has_variance(y_pred):
        return True
    if boundary <= 0 or boundary >= len(y_pred):
        return False
    ordered = sorted(y_pred)
    return abs(ordered[boundary] - ordered[boundary - 1]) <= PREDICTION_VARIANCE_EPSILON


def _row_count_boundaries_unreliable(
    y_pred: list[float],
    *,
    bins: int,
    top_only: bool = False,
) -> bool:
    if not _prediction_has_variance(y_pred):
        return True
    if bins <= 1 or len(y_pred) < 2:
        return True
    if not top_only and len(y_pred) < bins:
        return True
    boundary_indexes = [bins - 1] if top_only else list(range(1, bins))
    return any(
        _row_count_boundary_unreliable(
            y_pred,
            math.floor(boundary_idx * len(y_pred) / bins),
        )
        for boundary_idx in boundary_indexes
    )


def _weighted_boundaries_unreliable(
    y_pred: list[float],
    weights: list[float],
    *,
    bins: int,
    top_only: bool = False,
) -> bool:
    if not _prediction_has_variance(y_pred):
        return True
    total_weight = sum(weights)
    if bins <= 1 or total_weight <= 0:
        return True
    if not top_only and len(weights) < bins:
        return True
    target_weight = total_weight / bins
    boundaries = (
        [total_weight * (bins - 1) / bins]
        if top_only else [total_weight * idx / bins for idx in range(1, bins)]
    )
    ordered = sorted(zip(y_pred, weights), key=lambda row: row[0])
    cumulative = 0.0
    idx = 0
    while idx < len(ordered):
        end = idx + 1
        group_weight = ordered[idx][1]
        while (
            end < len(ordered)
            and abs(ordered[end][0] - ordered[end - 1][0])
            <= PREDICTION_VARIANCE_EPSILON
        ):
            group_weight += ordered[end][1]
            end += 1
        next_cumulative = cumulative + group_weight
        for boundary in boundaries:
            boundary_inside_group = (
                cumulative + WEIGHT_BOUNDARY_EPSILON
                < boundary
                < next_cumulative - WEIGHT_BOUNDARY_EPSILON
            )
            if not boundary_inside_group:
                continue
            if end - idx > 1:
                return True
            if group_weight > target_weight + WEIGHT_BOUNDARY_EPSILON:
                return True
        cumulative = next_cumulative
        idx = end
    return False


def _prediction_bins(
    y_true: list[float],
    y_pred: list[float],
    *,
    bins: int,
    weights: list[float] | None = None,
) -> list[list[float]]:
    if bins <= 0:
        return []
    ranked = _ranked_prediction_rows(y_true, y_pred, weights)
    count = len(ranked)
    out: list[list[float]] = []
    for idx in range(bins):
        start = math.floor(idx * count / bins)
        end = math.floor((idx + 1) * count / bins)
        out.append([label for _pred, label, _weight in ranked[start:end]])
    return out


def _prediction_weight_bins(
    y_true: list[float],
    y_pred: list[float],
    weights: list[float] | None,
    *,
    bins: int,
) -> list[list[tuple[float, float]]]:
    _assert_metric_inputs(y_true, y_pred, weights)
    if bins <= 0:
        return []
    ranked = _ranked_prediction_rows(y_true, y_pred, weights)
    total_weight = sum(weight for _pred, _label, weight in ranked)
    out: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    if total_weight <= 0:
        return out
    prefix_weights = [0.0]
    for _pred, _label, weight in ranked:
        prefix_weights.append(prefix_weights[-1] + weight)
    boundaries = [0]
    count = len(ranked)
    previous_boundary = 0
    for boundary_idx in range(1, bins):
        target_weight = total_weight * boundary_idx / bins
        if count >= bins:
            min_idx = max(previous_boundary + 1, boundary_idx)
            max_idx = count - (bins - boundary_idx)
        else:
            min_idx = previous_boundary
            max_idx = count
        if min_idx > max_idx:
            split_idx = previous_boundary
        else:
            split_idx = min(
                range(min_idx, max_idx + 1),
                key=lambda idx: (
                    abs(prefix_weights[idx] - target_weight),
                    idx,
                ),
            )
        boundaries.append(split_idx)
        previous_boundary = split_idx
    boundaries.append(count)
    for bin_idx in range(bins):
        start = boundaries[bin_idx]
        end = boundaries[bin_idx + 1]
        out[bin_idx] = [
            (label, weight)
            for _pred, label, weight in ranked[start:end]
        ]
    return out


def _weighted_bucket_mean(bucket: list[tuple[float, float]]) -> float:
    return _weighted_mean(
        [label for label, _weight in bucket],
        [weight for _label, weight in bucket],
    )


def _weighted_bucket_win_rate(bucket: list[tuple[float, float]]) -> float:
    return _weighted_win_rate(
        [label for label, _weight in bucket],
        [weight for _label, weight in bucket],
    )


def _bucket_weight(bucket: list[tuple[float, float]]) -> float:
    return sum(weight for _label, weight in bucket)


def _prediction_quality_metrics(
    y_true: list[float],
    y_pred: list[float],
    *,
    weights: list[float] | None = None,
    small_fold_threshold: int = 30,
) -> dict[str, Any]:
    _assert_metric_inputs(y_true, y_pred, weights)
    value_ic = _pearson(y_true, y_pred)
    rank_ic = _spearman(y_true, y_pred)
    population_mean = _mean(y_true)
    population_win_rate = _win_rate(y_true)
    row_weights = weights or [1.0 for _ in y_true]
    weighted_population_mean = _weighted_mean(y_true, row_weights)
    weighted_population_win_rate = _weighted_win_rate(y_true, row_weights)
    decile_buckets = _prediction_bins(y_true, y_pred, bins=10)
    per_decile_unreliable = _row_count_boundaries_unreliable(y_pred, bins=10)
    per_decile = (
        [math.nan for _ in range(10)]
        if per_decile_unreliable
        else [_mean(bucket) for bucket in decile_buckets]
    )
    weighted_decile_buckets = _prediction_weight_bins(y_true, y_pred, row_weights, bins=10)
    weighted_per_decile_unreliable = _weighted_boundaries_unreliable(
        y_pred,
        row_weights,
        bins=10,
    )
    weighted_per_decile = (
        [math.nan for _ in range(10)]
        if weighted_per_decile_unreliable
        else [_weighted_bucket_mean(bucket) for bucket in weighted_decile_buckets]
    )
    too_small = len(y_true) < small_fold_threshold
    effective_bins = 5 if too_small else 10
    effective_quantile = 1.0 / effective_bins if effective_bins else math.nan
    effective_bucket_name = "top_quintile" if effective_bins == 5 else "top_decile"
    top_decile_unreliable = too_small or _row_count_boundaries_unreliable(
        y_pred,
        bins=10,
        top_only=True,
    )
    weighted_top_decile_unreliable = too_small or _weighted_boundaries_unreliable(
        y_pred,
        row_weights,
        bins=10,
        top_only=True,
    )
    top_effective_unreliable = _row_count_boundaries_unreliable(
        y_pred,
        bins=effective_bins,
        top_only=True,
    )
    weighted_top_effective_unreliable = _weighted_boundaries_unreliable(
        y_pred,
        row_weights,
        bins=effective_bins,
        top_only=True,
    )
    effective_buckets = _prediction_bins(y_true, y_pred, bins=effective_bins)
    effective_weight_buckets = _prediction_weight_bins(
        y_true,
        y_pred,
        row_weights,
        bins=effective_bins,
    )
    top_decile_bucket = decile_buckets[-1] if decile_buckets else []
    top_decile_weight_bucket = weighted_decile_buckets[-1] if weighted_decile_buckets else []
    top_effective_bucket = effective_buckets[-1] if effective_buckets else []
    top_effective_weight_bucket = (
        effective_weight_buckets[-1] if effective_weight_buckets else []
    )
    total_weight = sum(row_weights)
    top_decile_weight_share = (
        _bucket_weight(top_decile_weight_bucket) / total_weight
        if total_weight > 0 else math.nan
    )
    top_effective_weight_share = (
        _bucket_weight(top_effective_weight_bucket) / total_weight
        if total_weight > 0 else math.nan
    )
    decile_rest = [
        label
        for bucket in decile_buckets[:-1]
        for label in bucket
    ]
    decile_rest_weight_bucket = [
        item
        for bucket in weighted_decile_buckets[:-1]
        for item in bucket
    ]
    effective_rest = [
        label
        for bucket in effective_buckets[:-1]
        for label in bucket
    ]
    effective_rest_weight_bucket = [
        item
        for bucket in effective_weight_buckets[:-1]
        for item in bucket
    ]
    top_decile_mean = (
        math.nan if top_decile_unreliable else _mean(top_decile_bucket)
    )
    weighted_top_decile_mean = (
        math.nan
        if weighted_top_decile_unreliable
        else _weighted_bucket_mean(top_decile_weight_bucket)
    )
    top_decile_bottom_mean = (
        math.nan if top_decile_unreliable else _mean(decile_rest)
    )
    weighted_top_decile_bottom_mean = (
        math.nan
        if weighted_top_decile_unreliable
        else _weighted_bucket_mean(decile_rest_weight_bucket)
    )
    top_effective_mean = (
        math.nan if top_effective_unreliable else _mean(top_effective_bucket)
    )
    weighted_top_effective_mean = (
        math.nan
        if weighted_top_effective_unreliable
        else _weighted_bucket_mean(top_effective_weight_bucket)
    )
    top_effective_bottom_mean = (
        math.nan if top_effective_unreliable else _mean(effective_rest)
    )
    weighted_top_effective_bottom_mean = (
        math.nan
        if weighted_top_effective_unreliable
        else _weighted_bucket_mean(effective_rest_weight_bucket)
    )
    lift_denominator_unreliable = (
        population_mean <= LIFT_POPULATION_MEAN_FLOOR
        or math.isnan(population_mean)
    )
    weighted_lift_denominator_unreliable = (
        weighted_population_mean <= LIFT_POPULATION_MEAN_FLOOR
        or math.isnan(weighted_population_mean)
    )
    if lift_denominator_unreliable:
        top_decile_lift = math.nan
        top_effective_lift = math.nan
    else:
        top_decile_lift = top_decile_mean / population_mean
        top_effective_lift = top_effective_mean / population_mean
    if weighted_lift_denominator_unreliable:
        weighted_top_decile_lift = math.nan
        weighted_top_effective_lift = math.nan
    else:
        weighted_top_decile_lift = weighted_top_decile_mean / weighted_population_mean
        weighted_top_effective_lift = weighted_top_effective_mean / weighted_population_mean
    top_decile_spread = (
        top_decile_mean - top_decile_bottom_mean
        if not math.isnan(top_decile_mean) and not math.isnan(top_decile_bottom_mean)
        else math.nan
    )
    weighted_top_decile_spread = (
        weighted_top_decile_mean - weighted_top_decile_bottom_mean
        if not math.isnan(weighted_top_decile_mean)
        and not math.isnan(weighted_top_decile_bottom_mean)
        else math.nan
    )
    top_effective_spread = (
        top_effective_mean - top_effective_bottom_mean
        if not math.isnan(top_effective_mean)
        and not math.isnan(top_effective_bottom_mean)
        else math.nan
    )
    weighted_top_effective_spread = (
        weighted_top_effective_mean - weighted_top_effective_bottom_mean
        if not math.isnan(weighted_top_effective_mean)
        and not math.isnan(weighted_top_effective_bottom_mean)
        else math.nan
    )
    return {
        "value_ic": value_ic,
        "rank_ic": rank_ic,
        "population_mean_label": population_mean,
        "population_win_rate": population_win_rate,
        "top_decile_lift": top_decile_lift,
        "top_decile_spread": top_decile_spread,
        "top_decile_mean_label": top_decile_mean,
        "top_decile_median_label": (
            math.nan if top_decile_unreliable else _median(top_decile_bucket)
        ),
        "top_decile_win_rate": (
            math.nan if top_decile_unreliable else _win_rate(top_decile_bucket)
        ),
        "bottom_9_deciles_mean_label": top_decile_bottom_mean,
        "top_effective_lift": top_effective_lift,
        "top_effective_spread": top_effective_spread,
        "top_effective_mean_label": top_effective_mean,
        "top_effective_win_rate": (
            math.nan if top_effective_unreliable else _win_rate(top_effective_bucket)
        ),
        "top_effective_quantile": effective_quantile,
        "top_effective_bins": effective_bins,
        "top_effective_bucket_name": effective_bucket_name,
        "per_decile_mean_label": per_decile,
        "weighted_population_mean_label": weighted_population_mean,
        "weighted_population_win_rate": weighted_population_win_rate,
        "weighted_top_decile_lift": weighted_top_decile_lift,
        "weighted_top_decile_spread": weighted_top_decile_spread,
        "weighted_top_decile_mean_label": weighted_top_decile_mean,
        "weighted_top_decile_win_rate": (
            math.nan
            if weighted_top_decile_unreliable
            else _weighted_bucket_win_rate(top_decile_weight_bucket)
        ),
        "weighted_bottom_9_deciles_mean_label": weighted_top_decile_bottom_mean,
        "weighted_top_effective_lift": weighted_top_effective_lift,
        "weighted_top_effective_spread": weighted_top_effective_spread,
        "weighted_top_effective_mean_label": weighted_top_effective_mean,
        "weighted_top_effective_win_rate": (
            math.nan
            if weighted_top_effective_unreliable
            else _weighted_bucket_win_rate(top_effective_weight_bucket)
        ),
        "weighted_top_effective_weight_share": (
            math.nan
            if weighted_top_effective_unreliable
            else top_effective_weight_share
        ),
        "weighted_per_decile_mean_label": weighted_per_decile,
        "weighted_top_decile_weight_share": (
            math.nan
            if weighted_top_decile_unreliable
            else top_decile_weight_share
        ),
        "per_decile_unreliable": per_decile_unreliable,
        "weighted_per_decile_unreliable": weighted_per_decile_unreliable,
        "top_decile_unreliable": top_decile_unreliable,
        "weighted_top_decile_unreliable": weighted_top_decile_unreliable,
        "top_effective_unreliable": top_effective_unreliable,
        "weighted_top_effective_unreliable": weighted_top_effective_unreliable,
        "top_decile_lift_unreliable": lift_denominator_unreliable,
        "top_effective_lift_unreliable": lift_denominator_unreliable,
        "weighted_top_decile_lift_unreliable": weighted_lift_denominator_unreliable,
        "weighted_top_effective_lift_unreliable": weighted_lift_denominator_unreliable,
        "top_decile_too_small": too_small,
        "top_decile_effective_bins": effective_bins,
        "top_decile_effective_quantile": effective_quantile,
        "top_decile_fallback": "quintile" if too_small else None,
    }


def _mean_metric(values: list[Any]) -> Any:
    if not values:
        return math.nan
    if all(isinstance(value, (int, float)) for value in values):
        finite = [float(value) for value in values if math.isfinite(float(value))]
        return _mean(finite)
    if all(isinstance(value, list) for value in values):
        width = max((len(value) for value in values), default=0)
        out: list[float] = []
        for idx in range(width):
            bucket = [
                float(value[idx])
                for value in values
                if idx < len(value)
                and isinstance(value[idx], (int, float))
                and math.isfinite(float(value[idx]))
            ]
            out.append(_mean(bucket))
        return out
    return None


def _mean_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    unweighted_top_scalar_fields = (
        "top_decile_lift",
        "top_decile_spread",
        "top_decile_mean_label",
        "top_decile_median_label",
        "top_decile_win_rate",
        "bottom_9_deciles_mean_label",
    )
    unweighted_top_effective_scalar_fields = (
        "top_effective_lift",
        "top_effective_spread",
        "top_effective_mean_label",
        "top_effective_win_rate",
    )
    weighted_top_scalar_fields = (
        "weighted_top_decile_lift",
        "weighted_top_decile_spread",
        "weighted_top_decile_mean_label",
        "weighted_top_decile_win_rate",
        "weighted_top_decile_weight_share",
        "weighted_bottom_9_deciles_mean_label",
    )
    weighted_top_effective_scalar_fields = (
        "weighted_top_effective_lift",
        "weighted_top_effective_spread",
        "weighted_top_effective_mean_label",
        "weighted_top_effective_win_rate",
        "weighted_top_effective_weight_share",
    )
    fields = (
        "weighted_mae",
        "pearson",
        "value_ic",
        "rank_ic",
        "top_quantile_label_mean",
        "population_mean_label",
        "population_win_rate",
        "top_decile_lift",
        "top_decile_spread",
        "top_decile_mean_label",
        "top_decile_median_label",
        "top_decile_win_rate",
        "bottom_9_deciles_mean_label",
        "top_effective_lift",
        "top_effective_spread",
        "top_effective_mean_label",
        "top_effective_win_rate",
        "per_decile_mean_label",
        "weighted_population_mean_label",
        "weighted_population_win_rate",
        "weighted_top_decile_lift",
        "weighted_top_decile_spread",
        "weighted_top_decile_mean_label",
        "weighted_top_decile_win_rate",
        "weighted_bottom_9_deciles_mean_label",
        "weighted_per_decile_mean_label",
        "weighted_top_decile_weight_share",
        "weighted_top_effective_lift",
        "weighted_top_effective_spread",
        "weighted_top_effective_mean_label",
        "weighted_top_effective_win_rate",
        "weighted_top_effective_weight_share",
    )
    out = {field: _mean_metric([metrics.get(field) for metrics in fold_metrics]) for field in fields}
    out["fold_count"] = len(fold_metrics)
    out["too_small_fold_count"] = sum(1 for metrics in fold_metrics if metrics.get("top_decile_too_small"))
    out["unreliable_top_decile_fold_count"] = sum(
        1 for metrics in fold_metrics if metrics.get("top_decile_unreliable")
    )
    out["unreliable_weighted_top_decile_fold_count"] = sum(
        1 for metrics in fold_metrics if metrics.get("weighted_top_decile_unreliable")
    )
    out["unreliable_top_effective_fold_count"] = sum(
        1 for metrics in fold_metrics if metrics.get("top_effective_unreliable")
    )
    out["unreliable_weighted_top_effective_fold_count"] = sum(
        1 for metrics in fold_metrics if metrics.get("weighted_top_effective_unreliable")
    )
    out["unreliable_top_quantile_fold_count"] = sum(
        1 for metrics in fold_metrics if metrics.get("top_quantile_unreliable")
    )
    out["unreliable_lift_fold_count"] = sum(
        1 for metrics in fold_metrics if metrics.get("top_decile_lift_unreliable")
    )
    out["unreliable_effective_lift_fold_count"] = sum(
        1 for metrics in fold_metrics if metrics.get("top_effective_lift_unreliable")
    )
    out["unreliable_weighted_lift_fold_count"] = sum(
        1
        for metrics in fold_metrics
        if metrics.get("weighted_top_decile_lift_unreliable")
    )
    out["unreliable_weighted_effective_lift_fold_count"] = sum(
        1
        for metrics in fold_metrics
        if metrics.get("weighted_top_effective_lift_unreliable")
    )
    if out["unreliable_top_quantile_fold_count"]:
        out["top_quantile_label_mean"] = math.nan
    if out["unreliable_lift_fold_count"]:
        out["top_decile_lift"] = math.nan
    if out["unreliable_effective_lift_fold_count"]:
        out["top_effective_lift"] = math.nan
    if out["unreliable_weighted_lift_fold_count"]:
        out["weighted_top_decile_lift"] = math.nan
    if out["unreliable_weighted_effective_lift_fold_count"]:
        out["weighted_top_effective_lift"] = math.nan
    if out["too_small_fold_count"] or out["unreliable_top_decile_fold_count"]:
        for field in unweighted_top_scalar_fields:
            out[field] = math.nan
    if out["too_small_fold_count"] or out["unreliable_weighted_top_decile_fold_count"]:
        for field in weighted_top_scalar_fields:
            out[field] = math.nan
    if out["unreliable_top_effective_fold_count"]:
        for field in unweighted_top_effective_scalar_fields:
            out[field] = math.nan
    if out["unreliable_weighted_top_effective_fold_count"]:
        for field in weighted_top_effective_scalar_fields:
            out[field] = math.nan
    out["unreliable_per_decile_fold_count"] = sum(
        1 for metrics in fold_metrics if metrics.get("per_decile_unreliable")
    )
    out["unreliable_weighted_per_decile_fold_count"] = sum(
        1 for metrics in fold_metrics if metrics.get("weighted_per_decile_unreliable")
    )
    if out["unreliable_per_decile_fold_count"]:
        width = max(
            (
                len(metrics.get("per_decile_mean_label", []))
                for metrics in fold_metrics
                if isinstance(metrics.get("per_decile_mean_label"), list)
            ),
            default=10,
        )
        out["per_decile_mean_label"] = [math.nan for _ in range(width)]
    if out["unreliable_weighted_per_decile_fold_count"]:
        width = max(
            (
                len(metrics.get("weighted_per_decile_mean_label", []))
                for metrics in fold_metrics
                if isinstance(metrics.get("weighted_per_decile_mean_label"), list)
            ),
            default=10,
        )
        out["weighted_per_decile_mean_label"] = [math.nan for _ in range(width)]
    return out


def _bool_model_param(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _resolved_model_params(pattern: PatternManifest) -> dict[str, Any]:
    raw = dict(pattern.model_params or {})
    min_samples_leaf = int(raw.get("min_samples_leaf", DEFAULT_MIN_SAMPLES_LEAF))
    if min_samples_leaf <= 1:
        LOGGER.warning(
            "Stage-1 trainer manifest set min_samples_leaf=%s; production "
            "default is %s",
            min_samples_leaf,
            DEFAULT_MIN_SAMPLES_LEAF,
        )
    return {
        "min_samples_leaf": min_samples_leaf,
        "l2_regularization": float(
            raw.get("l2_regularization", DEFAULT_L2_REGULARIZATION)
        ),
        "early_stopping": _bool_model_param(
            raw.get("early_stopping"),
            default=DEFAULT_EARLY_STOPPING,
        ),
        "min_cv_train_rows": int(
            raw.get("min_cv_train_rows", DEFAULT_MIN_CV_TRAIN_ROWS)
        ),
        "min_cv_train_securities": int(
            raw.get("min_cv_train_securities", DEFAULT_MIN_CV_TRAIN_SECURITIES)
        ),
    }


def _new_gbrt_model(
    *,
    max_iter: int,
    random_state: int,
    model_params: dict[str, Any] | None = None,
) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    params = dict(model_params or {})
    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        min_samples_leaf=int(params.get("min_samples_leaf", DEFAULT_MIN_SAMPLES_LEAF)),
        l2_regularization=float(
            params.get("l2_regularization", DEFAULT_L2_REGULARIZATION)
        ),
        early_stopping=_bool_model_param(
            params.get("early_stopping"),
            default=DEFAULT_EARLY_STOPPING,
        ),
        random_state=random_state,
    )


def _strict_json_ready(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _strict_json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_ready(child) for child in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _strict_json_dumps(payload: Any) -> str:
    return json.dumps(
        _strict_json_ready(payload),
        sort_keys=True,
        allow_nan=False,
    )


def _matrix(examples: list[TrainingExample]) -> list[list[float]]:
    return [row.vector.values for row in examples]


def _labels(examples: list[TrainingExample]) -> list[float]:
    return [row.label for row in examples]


def _cv_examples(examples: list[TrainingExample]) -> list[CVExample]:
    return [
        CVExample(
            signal_id=row.signal_id,
            ticker=row.ticker,
            security_identity=row.security_identity,
            signal_date=row.signal_date,
        )
        for row in examples
    ]


def _assert_single_observation_horizon(
    rows: list[ForwardReturnObservation],
    *,
    expected_horizon: str,
) -> None:
    horizons = {str(row.signal_horizon) for row in rows if row.signal_horizon}
    if len(horizons) > 1:
        raise RuntimeError(
            "Stage-1 training examples contain mixed signal_horizon values: "
            f"{sorted(horizons)}"
        )
    if expected_horizon and horizons and horizons != {expected_horizon}:
        raise RuntimeError(
            "Stage-1 training examples do not match manifest signal_horizon "
            f"{expected_horizon!r}: {sorted(horizons)}"
        )


def _row_direction(row: ForwardReturnObservation) -> str:
    direction = str(row.direction or getattr(row.signal, "direction", "") or "").lower()
    if direction not in {"long", "short"}:
        raise RuntimeError(
            "Stage-1 training row has invalid direction "
            f"{direction!r} for signal_id={row.signal_id!r}"
        )
    signal_direction = str(getattr(row.signal, "direction", "") or "").lower()
    if signal_direction and signal_direction != direction:
        raise RuntimeError(
            "Stage-1 training row direction does not match signal direction "
            f"for signal_id={row.signal_id!r}: "
            f"fro={direction!r} signal={signal_direction!r}"
        )
    return direction


def _assert_single_manifest_direction(
    rows: list[ForwardReturnObservation],
    *,
    expected_direction: str,
) -> str | None:
    directions = {_row_direction(row) for row in rows}
    if len(directions) > 1:
        raise RuntimeError(
            "Stage-1 training examples contain mixed directions: "
            f"{sorted(directions)}"
        )
    if not directions:
        return None
    selected_direction = next(iter(directions))
    if selected_direction != expected_direction:
        raise RuntimeError(
            "Stage-1 training examples do not match manifest direction "
            f"{expected_direction!r}: selected {selected_direction!r}"
        )
    return selected_direction


def _direction_signed_label(raw_label: float, direction: str) -> float:
    return raw_label if direction == "long" else -raw_label


def _realized_label_window_sessions(row: ForwardReturnObservation) -> int:
    entry = _parse_date(row.entry_session_date)
    exit_ = _parse_date(row.exit_session_date)
    if entry is None or exit_ is None:
        raise RuntimeError(
            "Stage-1 training row is missing realized label session dates: "
            f"signal_id={row.signal_id!r} "
            f"entry_session_date={row.entry_session_date!r} "
            f"exit_session_date={row.exit_session_date!r}"
        )
    if not is_us_equity_session(entry) or not is_us_equity_session(exit_):
        raise RuntimeError(
            "Stage-1 training row has non-session realized label dates: "
            f"signal_id={row.signal_id!r} "
            f"entry_session_date={entry.isoformat()} "
            f"exit_session_date={exit_.isoformat()}"
        )
    if exit_ < entry:
        raise RuntimeError(
            "Stage-1 training row has an exit session before entry session: "
            f"signal_id={row.signal_id!r} "
            f"entry_session_date={entry.isoformat()} "
            f"exit_session_date={exit_.isoformat()}"
        )
    sessions = 0
    cursor = next_us_equity_session(entry + timedelta(days=1))
    while cursor <= exit_:
        sessions += 1
        cursor = next_us_equity_session(cursor + timedelta(days=1))
    return sessions


def _validate_realized_label_windows(
    rows: list[ForwardReturnObservation],
    *,
    declared_horizon_sessions: int,
) -> int | None:
    realized = [_realized_label_window_sessions(row) for row in rows]
    if not realized:
        return None
    max_window = max(realized)
    if max_window > declared_horizon_sessions:
        raise RuntimeError(
            "Stage-1 CV purge horizon is smaller than the realized label window: "
            f"declared_horizon_sessions={declared_horizon_sessions} "
            f"max_realized_label_window_sessions={max_window}"
        )
    return max_window


def _fallback_security_identity(ticker: str) -> ResolvedSecurityIdentity:
    normalized = str(ticker).upper()
    return ResolvedSecurityIdentity(
        security_identity=f"ticker:{normalized}",
        canonical_ticker=normalized,
    )


def _security_identity_for_row(
    row: ForwardReturnObservation,
    identities: dict[str, ResolvedSecurityIdentity],
) -> ResolvedSecurityIdentity:
    ticker = str(row.ticker or "").upper()
    return identities.get(ticker) or _fallback_security_identity(ticker)


def _dedupe_training_candidates(
    candidates: list[tuple[ForwardReturnObservation, TrainingExample]],
    identities: dict[str, ResolvedSecurityIdentity],
) -> tuple[list[TrainingExample], int]:
    selected: dict[tuple[str, date], tuple[ForwardReturnObservation, TrainingExample]] = {}

    def priority(row: ForwardReturnObservation) -> tuple[
        bool,
        datetime,
        datetime,
        str,
        str,
        str,
    ]:
        identity = _security_identity_for_row(row, identities)
        return (
            str(row.ticker or "").upper() == identity.canonical_ticker,
            row.updated_at or datetime.min.replace(tzinfo=timezone.utc),
            row.created_at or datetime.min.replace(tzinfo=timezone.utc),
            identity.security_identity,
            str(row.ticker or "").upper(),
            str(row.forward_return_observation_id),
        )

    for row, example in candidates:
        identity = _security_identity_for_row(row, identities)
        key = (identity.security_identity, _signal_date(row.signal))
        current = selected.get(key)
        if current is None or priority(row) > priority(current[0]):
            selected[key] = (row, example)
    dropped = len(candidates) - len(selected)
    if dropped:
        LOGGER.warning(
            "Stage-1 trainer dropped %s duplicate finite rows after security "
            "identity collapse",
            dropped,
        )
    return [example for _row, example in selected.values()], dropped


def _apply_counted_hard_filter(
    query: Any,
    condition: Any,
    *,
    name: str,
    drop_counts: dict[str, int],
) -> Any:
    before = query.count()
    filtered = query.filter(condition)
    after = filtered.count()
    drop_counts[name] = int(before - after)
    return filtered


def _mean_or_nan(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _win_rate_or_nan(values: list[float]) -> float:
    return sum(1 for value in values if value > 0.0) / len(values) if values else math.nan


def _drop_accounting_metrics(
    *,
    kept_labels: list[float],
    dropped_labels: list[float],
    selection: dict[str, Any],
) -> dict[str, Any]:
    total = len(kept_labels) + len(dropped_labels)
    kept_mean = _mean_or_nan(kept_labels)
    dropped_mean = _mean_or_nan(dropped_labels)
    kept_win_rate = _win_rate_or_nan(kept_labels)
    dropped_win_rate = _win_rate_or_nan(dropped_labels)
    mean_delta = (
        dropped_mean - kept_mean
        if math.isfinite(dropped_mean) and math.isfinite(kept_mean)
        else math.nan
    )
    win_rate_delta = (
        dropped_win_rate - kept_win_rate
        if math.isfinite(dropped_win_rate) and math.isfinite(kept_win_rate)
        else math.nan
    )
    fraction = len(dropped_labels) / total if total else 0.0
    fraction_threshold = float(
        selection.get(
            "non_finite_drop_fraction_flag_threshold",
            DEFAULT_DROP_FRACTION_FLAG_THRESHOLD,
        )
    )
    mean_delta_threshold = float(
        selection.get(
            "non_finite_drop_label_mean_delta_threshold",
            DEFAULT_DROP_LABEL_MEAN_DELTA_THRESHOLD,
        )
    )
    bias_flag = (
        fraction >= fraction_threshold
        and math.isfinite(mean_delta)
        and abs(mean_delta) >= mean_delta_threshold
    )
    return {
        "selected_row_count": total,
        "kept_row_count": len(kept_labels),
        "dropped_non_finite_fraction": fraction,
        "kept_label_mean": kept_mean,
        "kept_label_win_rate": kept_win_rate,
        "dropped_non_finite_label_mean": dropped_mean,
        "dropped_non_finite_label_win_rate": dropped_win_rate,
        "dropped_non_finite_label_mean_delta": mean_delta,
        "dropped_non_finite_win_rate_delta": win_rate_delta,
        "dropped_non_finite_selection_bias_flag": bias_flag,
        "drop_fraction_flag_threshold": fraction_threshold,
        "drop_label_mean_delta_threshold": mean_delta_threshold,
    }


def _canonical_observation_rows(
    rows: list[ForwardReturnObservation],
) -> list[ForwardReturnObservation]:
    out: list[ForwardReturnObservation] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        if row.signal_id in seen:
            duplicates.add(row.signal_id)
            continue
        seen.add(row.signal_id)
        out.append(row)
    if duplicates:
        LOGGER.warning(
            "Stage-1 trainer selected latest canonical forward-return rows for "
            "%s signals with duplicate observations",
            len(duplicates),
        )
    return out


def _preload_feature_snapshots(
    session: Session,
    rows: list[ForwardReturnObservation],
) -> None:
    """Warm the session identity map for feature snapshots used by the cohort."""

    snapshot_ids = sorted(
        {
            str(row.signal.feature_snapshot_id)
            for row in rows
            if row.signal is not None and row.signal.feature_snapshot_id
        }
    )
    for start in range(0, len(snapshot_ids), FEATURE_SNAPSHOT_PRELOAD_CHUNK_SIZE):
        chunk = snapshot_ids[start : start + FEATURE_SNAPSHOT_PRELOAD_CHUNK_SIZE]
        session.query(FeatureSnapshot).filter(
            FeatureSnapshot.feature_snapshot_id.in_(chunk)
        ).all()


def _load_training_examples(
    session: Session,
    *,
    pattern: PatternManifest,
    return_metrics: bool = False,
) -> list[TrainingExample] | tuple[list[TrainingExample], dict[str, Any]]:
    selection = pattern.selection
    if selection.get("source") != "forward_return_observations":
        raise RuntimeError(
            "Stage-1 train_model currently supports manifest selection source "
            "'forward_return_observations' only"
        )
    feature_schema = dict(pattern.feature_schema)
    feature_schema.setdefault("pattern_id", pattern.pattern_id)
    audit_feature_schema_no_leakage(feature_schema)

    label_field = str(pattern.label.get("field") or "forward_return")
    if label_field.lower().startswith(("forward_path", "feature_")):
        raise RuntimeError(f"unsupported label field {label_field!r}")
    raw_statuses = selection.get("statuses") or ["computed"]
    statuses = [raw_statuses] if isinstance(raw_statuses, str) else list(raw_statuses)
    query = (
        session.query(ForwardReturnObservation)
        .options(joinedload(ForwardReturnObservation.signal))
        .join(SignalRegistry, ForwardReturnObservation.signal_id == SignalRegistry.signal_id)
        .filter(ForwardReturnObservation.pattern_id == pattern.pattern_id)
        .filter(ForwardReturnObservation.status.in_(list(statuses)))
    )
    start = _parse_date(selection.get("start_date"))
    end = _parse_date(selection.get("end_date"))
    if start is not None:
        query = query.filter(SignalRegistry.signal_timestamp >= datetime.combine(start, datetime.min.time()).replace(tzinfo=timezone.utc))
    if end is not None:
        query = query.filter(SignalRegistry.signal_timestamp <= datetime.combine(end, datetime.max.time()).replace(tzinfo=timezone.utc))
    raw_signal_ids = selection.get("signal_ids")
    signal_ids = [raw_signal_ids] if isinstance(raw_signal_ids, str) else raw_signal_ids
    if signal_ids:
        query = query.filter(ForwardReturnObservation.signal_id.in_(list(signal_ids)))
    if pattern.signal_horizon:
        query = query.filter(
            ForwardReturnObservation.signal_horizon == pattern.signal_horizon
        )
    hard_filter_drop_counts: dict[str, int] = {}
    query = _apply_counted_hard_filter(
        query,
        SignalRegistry.signal_status == "active",
        name="signal_status_active",
        drop_counts=hard_filter_drop_counts,
    )
    query = _apply_counted_hard_filter(
        query,
        SignalRegistry.lookahead_guard_passed.is_(True),
        name="lookahead_guard_passed",
        drop_counts=hard_filter_drop_counts,
    )
    query = _apply_counted_hard_filter(
        query,
        SignalRegistry.forward_return_status == "computed",
        name="forward_return_status_computed",
        drop_counts=hard_filter_drop_counts,
    )
    pit_failed_row_count = query.filter(
        SignalRegistry.point_in_time_passed.isnot(True)
    ).count()
    if pit_failed_row_count and not pattern.allow_deferred_pit:
        raise RuntimeError(
            "Stage-1 trainer selected point-in-time failed rows; set "
            "selection.allow_deferred_pit=true only for the explicit I12 "
            "deferred-PIT rebuild path. "
            f"pit_failed_row_count={pit_failed_row_count}"
        )
    if pattern.allow_deferred_pit:
        hard_filter_drop_counts["point_in_time_passed"] = 0
        if pit_failed_row_count:
            LOGGER.warning(
                "Stage-1 trainer is using %s point-in-time failed rows because "
                "selection.allow_deferred_pit=true",
                pit_failed_row_count,
            )
    else:
        hard_filter_drop_counts["point_in_time_passed"] = int(pit_failed_row_count)
        query = query.filter(SignalRegistry.point_in_time_passed.is_(True))
    if any(hard_filter_drop_counts.values()):
        LOGGER.warning(
            "Stage-1 trainer hard-filter drop counts: %s",
            json.dumps(hard_filter_drop_counts, sort_keys=True),
        )
    rows = query.order_by(
        ForwardReturnObservation.signal_id.asc(),
        ForwardReturnObservation.updated_at.desc(),
        ForwardReturnObservation.created_at.desc(),
        ForwardReturnObservation.forward_return_observation_id.desc(),
    ).all()
    rows = _canonical_observation_rows(rows)
    _preload_feature_snapshots(session, rows)
    _assert_single_observation_horizon(
        rows,
        expected_horizon=pattern.signal_horizon,
    )
    expected_direction = str(pattern.direction or "long").lower()
    selected_direction = _assert_single_manifest_direction(
        rows,
        expected_direction=expected_direction,
    )
    declared_horizon_sessions = int(
        selection.get("horizon_sessions") or pattern.embargo_sessions
    )
    max_realized_label_window_sessions = _validate_realized_label_windows(
        rows,
        declared_horizon_sessions=declared_horizon_sessions,
    )
    identities = resolve_security_identities_for_tickers(
        session,
        [str(row.ticker or "") for row in rows],
    )
    rows = sorted(rows, key=lambda row: row.signal_timestamp)
    candidates: list[tuple[ForwardReturnObservation, TrainingExample]] = []
    dropped_non_finite = 0
    dropped_non_finite_by_feature: dict[str, int] = {}
    kept_labels: list[float] = []
    dropped_labels: list[float] = []
    for obs in rows:
        label_value = getattr(obs, label_field, None)
        if label_value is None:
            continue
        raw_label = float(label_value)
        if not math.isfinite(raw_label):
            raise RuntimeError(
                "Stage-1 training label is non-finite for "
                f"signal_id={obs.signal_id!r} field={label_field!r}"
            )
        direction = _row_direction(obs)
        label = _direction_signed_label(raw_label, direction)
        signal = obs.signal
        vector = select_features(session, obs.signal_id, feature_schema)
        corrupt_features = [
            name
            for name, status in vector.missing_statuses.items()
            if status == "non_finite_stored_value"
        ]
        if corrupt_features:
            dropped_non_finite += 1
            dropped_labels.append(label)
            for name in corrupt_features:
                dropped_non_finite_by_feature[name] = (
                    dropped_non_finite_by_feature.get(name, 0) + 1
                )
            continue
        kept_labels.append(label)
        identity = _security_identity_for_row(obs, identities)
        candidates.append(
            (
                obs,
                TrainingExample(
                    signal_id=obs.signal_id,
                    ticker=identity.canonical_ticker,
                    security_identity=identity.security_identity,
                    signal_date=_signal_date(signal),
                    label=label,
                    vector=vector,
                    direction=direction,
                    raw_label=raw_label,
                    realized_label_window_sessions=_realized_label_window_sessions(obs),
                ),
            )
        )
    out, security_identity_duplicate_rows_dropped = _dedupe_training_candidates(
        candidates,
        identities,
    )
    out = sorted(out, key=lambda row: (row.signal_date, row.signal_id))
    if dropped_non_finite:
        LOGGER.warning(
            "Stage-1 trainer dropped %s rows with non_finite_stored_value "
            "features: %s",
            dropped_non_finite,
            json.dumps(dropped_non_finite_by_feature, sort_keys=True),
        )
    drop_metrics = _drop_accounting_metrics(
        kept_labels=kept_labels,
        dropped_labels=dropped_labels,
        selection=selection,
    )
    if drop_metrics["dropped_non_finite_selection_bias_flag"] and not bool(
        selection.get("allow_skewed_non_finite_drops", False)
    ):
        raise RuntimeError(
            "Stage-1 trainer detected selection-biased non-finite feature drops: "
            f"dropped_non_finite_fraction={drop_metrics['dropped_non_finite_fraction']:.4f} "
            "dropped_non_finite_label_mean_delta="
            f"{drop_metrics['dropped_non_finite_label_mean_delta']:.6f}; "
            "set selection.allow_skewed_non_finite_drops=true only after review"
        )
    if return_metrics:
        return out, {
            "dropped_non_finite": dropped_non_finite,
            "dropped_non_finite_by_feature": dropped_non_finite_by_feature,
            **drop_metrics,
            "pit_deferred": bool(pattern.allow_deferred_pit and pit_failed_row_count),
            "pit_failed_row_count": int(pit_failed_row_count),
            "hard_filter_drop_counts": hard_filter_drop_counts,
            "security_identity_duplicate_rows_dropped": int(
                security_identity_duplicate_rows_dropped
            ),
            "expected_direction": expected_direction,
            "selected_direction": selected_direction,
            "declared_horizon_sessions": declared_horizon_sessions,
            "max_realized_label_window_sessions": max_realized_label_window_sessions,
        }
    return out


def _fold_train_weights(
    cv_rows: list[CVExample],
    train_indices: list[int],
) -> list[float]:
    return unique_name_weights([cv_rows[i] for i in train_indices])


def _fold_security_count(cv_rows: list[CVExample], indices: list[int]) -> int:
    return len({cv_rows[i].security_identity or cv_rows[i].ticker for i in indices})


def _cross_validate(
    examples: list[TrainingExample],
    *,
    horizon_sessions: int,
    embargo_sessions: int,
    n_splits: int,
    max_iter: int,
    random_state: int,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_model_params = dict(model_params or {})
    min_cv_train_rows = int(
        resolved_model_params.get("min_cv_train_rows", DEFAULT_MIN_CV_TRAIN_ROWS)
    )
    min_cv_train_securities = int(
        resolved_model_params.get(
            "min_cv_train_securities",
            DEFAULT_MIN_CV_TRAIN_SECURITIES,
        )
    )
    cv_rows = _cv_examples(examples)
    folds = purged_embargoed_walk_forward_splits(
        cv_rows,
        n_splits=n_splits,
        horizon_sessions=horizon_sessions,
        embargo_sessions=embargo_sessions,
    )
    fold_metrics: list[dict[str, Any]] = []
    all_true: list[float] = []
    all_pred: list[float] = []
    all_fold_percentiles: list[float] = []
    all_weights: list[float] = []
    dropped_nonviable_folds: list[dict[str, Any]] = []
    for idx, fold in enumerate(folds):
        train = [examples[i] for i in fold.train_indices]
        test = [examples[i] for i in fold.test_indices]
        train_security_count = _fold_security_count(cv_rows, fold.train_indices)
        nonviable_reasons: list[str] = []
        if len(train) < min_cv_train_rows:
            nonviable_reasons.append("train_count_below_minimum")
        if train_security_count < min_cv_train_securities:
            nonviable_reasons.append("train_security_count_below_minimum")
        if nonviable_reasons:
            dropped_nonviable_folds.append({
                "fold": idx,
                "train_count": len(train),
                "test_count": len(test),
                "train_security_count": train_security_count,
                "test_start_date": fold.test_start_date.isoformat(),
                "test_end_date": fold.test_end_date.isoformat(),
                "reasons": nonviable_reasons,
            })
            continue
        train_weights = _fold_train_weights(cv_rows, fold.train_indices)
        test_weights = unique_name_weights([cv_rows[i] for i in fold.test_indices])
        fold_model_params = dict(resolved_model_params)
        early_stopping_disabled_for_fold = False
        if (
            _bool_model_param(
                fold_model_params.get("early_stopping"),
                default=DEFAULT_EARLY_STOPPING,
            )
            and len(train) < DEFAULT_EARLY_STOPPING_MIN_TRAIN_ROWS
        ):
            fold_model_params["early_stopping"] = False
            early_stopping_disabled_for_fold = True
        model = _new_gbrt_model(
            max_iter=max_iter,
            random_state=random_state + idx,
            model_params=fold_model_params,
        )
        model.fit(
            _matrix(train),
            _labels(train),
            sample_weight=train_weights,
        )
        preds = [float(value) for value in model.predict(_matrix(test))]
        labels = _labels(test)
        all_true.extend(labels)
        all_pred.extend(preds)
        all_fold_percentiles.extend(_score_percentiles(preds))
        all_weights.extend(test_weights)
        value_ic = _pearson(labels, preds)
        fold_metrics.append(
            {
                "fold": idx,
                "train_count": len(train),
                "test_count": len(test),
                "train_security_count": train_security_count,
                "test_start_date": fold.test_start_date.isoformat(),
                "test_end_date": fold.test_end_date.isoformat(),
                "early_stopping_disabled_for_fold": early_stopping_disabled_for_fold,
                "weighted_mae": _weighted_mae(labels, preds, test_weights),
                "pearson": value_ic,
                "value_ic": value_ic,
                "top_quantile_label_mean": _top_quantile_mean(labels, preds),
                "top_quantile_unreliable": _top_quantile_unreliable(preds),
                **_prediction_quality_metrics(labels, preds, weights=test_weights),
            }
        )
    if not fold_metrics:
        raise RuntimeError(
            "Stage-1 CV has no viable folds after train-size/security-count "
            "guards"
        )
    raw_pooled_value_ic = _pearson(all_true, all_pred)
    raw_pooled_rank_ic = _spearman(all_true, all_pred)
    pooled_quality_metrics = _prediction_quality_metrics(
        all_true,
        all_fold_percentiles,
        weights=all_weights,
    )
    pooled_quality_metrics["pearson"] = pooled_quality_metrics["value_ic"]
    pooled_quality_metrics[
        "rank_ic_score_basis"
    ] = "fold_normalized_prediction_percentile_midrank"
    pooled_quality_metrics["pooled_score_basis"] = "fold_normalized_prediction_percentile"
    pooled_quality_metrics["raw_pooled_score_basis"] = "raw_prediction"
    pooled_quality_metrics["raw_pooled_pearson"] = raw_pooled_value_ic
    pooled_quality_metrics["raw_pooled_value_ic"] = raw_pooled_value_ic
    pooled_quality_metrics["raw_pooled_rank_ic"] = raw_pooled_rank_ic
    return {
        "cv_type": "purged_embargoed_walk_forward",
        "random_kfold_forbidden": True,
        "horizon_sessions": horizon_sessions,
        "embargo_sessions": embargo_sessions,
        "folds": fold_metrics,
        "dropped_nonviable_fold_count": len(dropped_nonviable_folds),
        "dropped_nonviable_folds": dropped_nonviable_folds,
        "min_cv_train_rows": min_cv_train_rows,
        "min_cv_train_securities": min_cv_train_securities,
        "mean_fold_metrics": _mean_fold_metrics(fold_metrics),
        "oos_count": len(all_true),
        **pooled_quality_metrics,
        "weighted_mae": _weighted_mae(all_true, all_pred, all_weights),
        "top_quantile_label_mean": _top_quantile_mean(all_true, all_fold_percentiles),
        "top_quantile_unreliable": _top_quantile_unreliable(all_fold_percentiles),
    }


def _quality_gate_config(pattern: PatternManifest) -> dict[str, Any]:
    config = dict(DEFAULT_OOS_GATE)
    config.update(pattern.oos_quality_gate or {})
    return {
        "min_top_decile_lift": float(config.get("min_top_decile_lift", 1.0)),
        "min_rank_ic": float(config.get("min_rank_ic", 0.0)),
        "required_metrics": list(
            config.get("required_metrics") or ["top_decile_lift", "rank_ic"]
        ),
        "reject_status": str(config.get("reject_status") or REJECTED_STATUS),
    }


def _finite_metric(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _evaluate_oos_quality_gate(
    cv_metrics: dict[str, Any],
    *,
    pattern: PatternManifest,
    pass_status: str,
) -> dict[str, Any]:
    config = _quality_gate_config(pattern)
    failures: list[dict[str, Any]] = []
    for metric_name in config["required_metrics"]:
        metric_value = cv_metrics.get(metric_name)
        if not _finite_metric(metric_value):
            failures.append(
                {
                    "metric": metric_name,
                    "reason": "non_finite_required_metric",
                    "value": metric_value,
                }
            )
    top_decile_lift = cv_metrics.get("top_decile_lift")
    if _finite_metric(top_decile_lift) and float(top_decile_lift) <= config["min_top_decile_lift"]:
        failures.append(
            {
                "metric": "top_decile_lift",
                "reason": "below_minimum",
                "value": top_decile_lift,
                "threshold": config["min_top_decile_lift"],
            }
        )
    rank_ic = cv_metrics.get("rank_ic")
    if _finite_metric(rank_ic) and float(rank_ic) <= config["min_rank_ic"]:
        failures.append(
            {
                "metric": "rank_ic",
                "reason": "below_minimum",
                "value": rank_ic,
                "threshold": config["min_rank_ic"],
            }
        )
    for flag in (
        "top_decile_unreliable",
        "top_decile_too_small",
        "top_decile_lift_unreliable",
    ):
        if cv_metrics.get(flag):
            failures.append({"metric": flag, "reason": "required_metric_flagged"})
    passed = not failures
    return {
        "passed": passed,
        "status": pass_status if passed else config["reject_status"],
        "status_on_pass": pass_status,
        "status_on_fail": config["reject_status"],
        "thresholds": {
            "min_top_decile_lift": config["min_top_decile_lift"],
            "min_rank_ic": config["min_rank_ic"],
            "required_metrics": config["required_metrics"],
        },
        "failures": failures,
    }


class Stage1TrainModelJob(BaseJob):
    def __init__(
        self,
        *,
        session: Session,
        manifest: FrozenMLManifest,
        pattern_id: str,
        artifact_dir: str | Path,
        status: str = DEFAULT_STATUS,
        n_splits: int = 3,
        max_iter: int = 80,
        random_state: int = 7,
    ) -> None:
        self.session = session
        self.manifest = manifest
        self.pattern_id = pattern_id
        self.artifact_dir = Path(artifact_dir)
        self.status = status
        self.n_splits = n_splits
        self.max_iter = max_iter
        self.random_state = random_state
        self.partial_metrics: dict[str, Any] = {}

    @property
    def job_name(self) -> str:
        return "stage1_train_model"

    @property
    def job_type(self) -> str:
        return "ml_training"

    def run(self, ctx: JobContext) -> JobResult:
        pattern = self.manifest.pattern(self.pattern_id)
        examples, selection_metrics = _load_training_examples(
            self.session,
            pattern=pattern,
            return_metrics=True,
        )
        graded_cohorts = {row.signal_date for row in examples}
        if len(graded_cohorts) < pattern.min_graded_cohorts:
            raise RuntimeError(
                f"pattern {self.pattern_id} has {len(graded_cohorts)} graded cohorts "
                f"across {len(examples)} rows; "
                f"minimum is {pattern.min_graded_cohorts}"
            )
        feature_schema = dict(pattern.feature_schema)
        feature_schema.setdefault("pattern_id", pattern.pattern_id)
        schema_hash = feature_schema_hash(feature_schema)
        horizon = int(pattern.selection.get("horizon_sessions") or pattern.embargo_sessions)
        model_params = _resolved_model_params(pattern)
        cv_metrics = _cross_validate(
            examples,
            horizon_sessions=horizon,
            embargo_sessions=pattern.embargo_sessions,
            n_splits=self.n_splits,
            max_iter=self.max_iter,
            random_state=self.random_state,
            model_params=model_params,
        )
        cv_metrics["signal_horizon"] = pattern.signal_horizon
        cv_metrics["training_selection"] = selection_metrics
        cv_metrics["model_params"] = model_params
        gate_decision = _evaluate_oos_quality_gate(
            cv_metrics,
            pattern=pattern,
            pass_status=self.status,
        )
        cv_metrics["oos_quality_gate"] = gate_decision
        registry_status = str(gate_decision["status"])
        weights = unique_name_weights(_cv_examples(examples))
        model = _new_gbrt_model(
            max_iter=self.max_iter,
            random_state=self.random_state,
            model_params=model_params,
        )
        matrix = _matrix(examples)
        model.fit(matrix, _labels(examples), sample_weight=weights)
        model_id = f"stage1_{self.pattern_id.lower()}_{schema_hash[:12]}_{uuid4().hex[:8]}"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.artifact_dir / f"{model_id}.pkl"
        training_params = {
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "n_splits": self.n_splits,
            "signal_horizon": pattern.signal_horizon,
            "horizon_sessions": horizon,
            "sample_weight": "unique_name_cluster_by_security_identity",
            "model_params": model_params,
        }
        artifact_payload = {
            "model_id": model_id,
            "model_family": MODEL_FAMILY,
            "pattern_id": self.pattern_id,
            "manifest_version": self.manifest.manifest_version,
            "manifest_sha256": self.manifest.manifest_sha256,
            "feature_schema": feature_schema,
            "feature_schema_hash": schema_hash,
            "feature_names": examples[0].vector.feature_names,
            "training_feature_ranges": _feature_ranges(matrix),
            "training_params": training_params,
            "model_params": model_params,
            "model": model,
        }
        with open(artifact_path, "wb") as f:
            pickle.dump(artifact_payload, f)
        signal_dates = [row.signal_date for row in examples]
        registry = MLModelRegistry(
            model_id=model_id,
            job_run_id=ctx.job_run_id,
            pattern_id=self.pattern_id,
            model_family=MODEL_FAMILY,
            training_window_start=min(signal_dates),
            training_window_end=max(signal_dates),
            manifest_version=self.manifest.manifest_version,
            manifest_sha256=self.manifest.manifest_sha256,
            feature_schema_hash=schema_hash,
            feature_code_git_sha=ctx.app_commit_sha or "unknown",
            training_params_json=_strict_json_dumps(training_params),
            cv_metrics_json=_strict_json_dumps(
                {
                    "per_pattern": {self.pattern_id: cv_metrics},
                    "pooled": {"diagnostic_only": True, "computed": False},
                },
            ),
            feature_schema_json=_strict_json_dumps(feature_schema),
            artifact_uri=str(artifact_path),
            status=registry_status,
        )
        self.session.add(registry)
        metrics = {
            "patterns_trained": [self.pattern_id],
            "training_rows": len(examples),
            "model_id": model_id,
            "artifact_uri": str(artifact_path),
            "feature_schema_hash": schema_hash,
            "cv_metrics": cv_metrics,
            "oos_quality_gate": gate_decision,
            "registry_status": registry_status,
            "pooled_metrics": {"diagnostic_only": True},
        }
        self.partial_metrics = metrics
        return JobResult(
            status="finished",
            metrics=metrics,
            input_hashes={"manifest_sha256": self.manifest.manifest_sha256},
            output_hashes={"feature_schema_hash": schema_hash},
            artifacts=[str(artifact_path)],
        )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--pattern-id", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--status", default=DEFAULT_STATUS)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument(
        "--db-statement-timeout-ms",
        type=int,
        default=DEFAULT_TRAINER_DB_TIMEOUT_MS,
        help=(
            "PostgreSQL statement_timeout for this offline trainer process; "
            "set before opening the DB session."
        ),
    )
    parser.add_argument(
        "--db-idle-in-transaction-timeout-ms",
        type=int,
        default=DEFAULT_TRAINER_DB_TIMEOUT_MS,
        help=(
            "PostgreSQL idle_in_transaction_session_timeout for this offline "
            "trainer process; set before opening the DB session."
        ),
    )
    return parser.parse_args(argv)


def _apply_trainer_db_timeout_env(args: argparse.Namespace) -> None:
    if args.db_statement_timeout_ms is not None:
        if args.db_statement_timeout_ms < 0:
            raise ValueError("--db-statement-timeout-ms must be >= 0")
        os.environ["ALPHA_DB_STATEMENT_TIMEOUT_MS"] = str(
            args.db_statement_timeout_ms
        )
    if args.db_idle_in_transaction_timeout_ms is not None:
        if args.db_idle_in_transaction_timeout_ms < 0:
            raise ValueError("--db-idle-in-transaction-timeout-ms must be >= 0")
        os.environ["ALPHA_DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS"] = str(
            args.db_idle_in_transaction_timeout_ms
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    _apply_trainer_db_timeout_env(args)
    session = get_session()
    manifest = load_manifest(args.manifest_path)
    job = Stage1TrainModelJob(
        session=session,
        manifest=manifest,
        pattern_id=args.pattern_id,
        artifact_dir=args.artifact_dir,
        status=args.status,
        n_splits=args.n_splits,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    result = run_job(
        session,
        job,
        params={
            "manifest_path": args.manifest_path,
            "pattern_id": args.pattern_id,
            "artifact_dir": args.artifact_dir,
            "status": args.status,
            "db_statement_timeout_ms": args.db_statement_timeout_ms,
            "db_idle_in_transaction_timeout_ms": (
                args.db_idle_in_transaction_timeout_ms
            ),
        },
    )
    print(_strict_json_dumps(result.metrics))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

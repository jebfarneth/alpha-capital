"""Train Stage-1 per-pattern ML rankers from frozen manifests.

The job is deliberately offline and shadow-first. It reads manifest-governed
signals, selects already-materialized stored features through
``alpha.ml.model_features``, trains one GBRT per pattern, writes a model
artifact, and registers that artifact for later shadow inference.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from alpha.db.engine import get_session
from alpha.db.models import (
    ForwardReturnObservation,
    MLModelRegistry,
    SignalRegistry,
)
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.runner import run_job
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


MODEL_FAMILY = "sklearn_hist_gradient_boosting_gbrt"
DEFAULT_STATUS = "shadow"


@dataclass(frozen=True)
class TrainingExample(CVExample):
    label: float
    vector: SelectedFeatureVector


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
    return [value for value in values if not math.isnan(value)]


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
    denom = sum(weights)
    if denom <= 0:
        return math.nan
    return sum(abs(a - b) * w for a, b, w in zip(y_true, y_pred, weights)) / denom


def _pearson(y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) < 2:
        return math.nan
    mean_true = sum(y_true) / len(y_true)
    mean_pred = sum(y_pred) / len(y_pred)
    num = sum((a - mean_true) * (b - mean_pred) for a, b in zip(y_true, y_pred))
    den_a = math.sqrt(sum((a - mean_true) ** 2 for a in y_true))
    den_b = math.sqrt(sum((b - mean_pred) ** 2 for b in y_pred))
    if den_a == 0 or den_b == 0:
        return math.nan
    return num / (den_a * den_b)


def _top_quantile_mean(
    y_true: list[float], y_pred: list[float], *, quantile: float = 0.2
) -> float:
    if not y_true:
        return math.nan
    count = max(1, int(math.ceil(len(y_true) * quantile)))
    ranked = sorted(zip(y_pred, y_true), key=lambda row: row[0], reverse=True)
    return sum(label for _, label in ranked[:count]) / count


def _new_gbrt_model(*, max_iter: int, random_state: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        min_samples_leaf=1,
        random_state=random_state,
    )


def _matrix(examples: list[TrainingExample]) -> list[list[float]]:
    return [row.vector.values for row in examples]


def _labels(examples: list[TrainingExample]) -> list[float]:
    return [row.label for row in examples]


def _cv_examples(examples: list[TrainingExample]) -> list[CVExample]:
    return [
        CVExample(signal_id=row.signal_id, ticker=row.ticker, signal_date=row.signal_date)
        for row in examples
    ]


def _load_training_examples(
    session: Session,
    *,
    pattern: PatternManifest,
) -> list[TrainingExample]:
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
    raw_statuses = selection.get("statuses") or ["computed", "finished"]
    statuses = [raw_statuses] if isinstance(raw_statuses, str) else list(raw_statuses)
    query = (
        session.query(ForwardReturnObservation)
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
    rows = query.order_by(ForwardReturnObservation.signal_timestamp.asc()).all()
    out: list[TrainingExample] = []
    for obs in rows:
        label_value = getattr(obs, label_field, None)
        if label_value is None:
            continue
        signal = obs.signal
        vector = select_features(session, obs.signal_id, feature_schema)
        out.append(
            TrainingExample(
                signal_id=obs.signal_id,
                ticker=obs.ticker,
                signal_date=_signal_date(signal),
                label=float(label_value),
                vector=vector,
            )
        )
    return out


def _cross_validate(
    examples: list[TrainingExample],
    *,
    horizon_sessions: int,
    embargo_sessions: int,
    n_splits: int,
    max_iter: int,
    random_state: int,
) -> dict[str, Any]:
    cv_rows = _cv_examples(examples)
    folds = purged_embargoed_walk_forward_splits(
        cv_rows,
        n_splits=n_splits,
        horizon_sessions=horizon_sessions,
        embargo_sessions=embargo_sessions,
    )
    weights = unique_name_weights(cv_rows)
    fold_metrics: list[dict[str, Any]] = []
    all_true: list[float] = []
    all_pred: list[float] = []
    all_weights: list[float] = []
    for idx, fold in enumerate(folds):
        train = [examples[i] for i in fold.train_indices]
        test = [examples[i] for i in fold.test_indices]
        model = _new_gbrt_model(max_iter=max_iter, random_state=random_state + idx)
        model.fit(
            _matrix(train),
            _labels(train),
            sample_weight=[weights[i] for i in fold.train_indices],
        )
        preds = [float(value) for value in model.predict(_matrix(test))]
        labels = _labels(test)
        test_weights = [weights[i] for i in fold.test_indices]
        all_true.extend(labels)
        all_pred.extend(preds)
        all_weights.extend(test_weights)
        fold_metrics.append(
            {
                "fold": idx,
                "train_count": len(train),
                "test_count": len(test),
                "test_start_date": fold.test_start_date.isoformat(),
                "test_end_date": fold.test_end_date.isoformat(),
                "weighted_mae": _weighted_mae(labels, preds, test_weights),
                "pearson": _pearson(labels, preds),
                "top_quantile_label_mean": _top_quantile_mean(labels, preds),
            }
        )
    return {
        "cv_type": "purged_embargoed_walk_forward",
        "random_kfold_forbidden": True,
        "horizon_sessions": horizon_sessions,
        "embargo_sessions": embargo_sessions,
        "folds": fold_metrics,
        "oos_count": len(all_true),
        "weighted_mae": _weighted_mae(all_true, all_pred, all_weights),
        "pearson": _pearson(all_true, all_pred),
        "top_quantile_label_mean": _top_quantile_mean(all_true, all_pred),
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
        examples = _load_training_examples(self.session, pattern=pattern)
        if len(examples) < pattern.min_graded_cohorts:
            raise RuntimeError(
                f"pattern {self.pattern_id} has {len(examples)} graded cohorts; "
                f"minimum is {pattern.min_graded_cohorts}"
            )
        feature_schema = dict(pattern.feature_schema)
        feature_schema.setdefault("pattern_id", pattern.pattern_id)
        schema_hash = feature_schema_hash(feature_schema)
        horizon = int(pattern.selection.get("horizon_sessions") or pattern.embargo_sessions)
        cv_metrics = _cross_validate(
            examples,
            horizon_sessions=horizon,
            embargo_sessions=pattern.embargo_sessions,
            n_splits=self.n_splits,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        weights = unique_name_weights(_cv_examples(examples))
        model = _new_gbrt_model(max_iter=self.max_iter, random_state=self.random_state)
        matrix = _matrix(examples)
        model.fit(matrix, _labels(examples), sample_weight=weights)
        model_id = f"stage1_{self.pattern_id.lower()}_{schema_hash[:12]}_{uuid4().hex[:8]}"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.artifact_dir / f"{model_id}.pkl"
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
            training_params_json=json.dumps(
                {
                    "max_iter": self.max_iter,
                    "random_state": self.random_state,
                    "n_splits": self.n_splits,
                    "sample_weight": "unique_name_cluster_by_ticker",
                },
                sort_keys=True,
            ),
            cv_metrics_json=json.dumps(
                {
                    "per_pattern": {self.pattern_id: cv_metrics},
                    "pooled": {"diagnostic_only": True, "computed": False},
                },
                sort_keys=True,
                default=str,
            ),
            feature_schema_json=json.dumps(feature_schema, sort_keys=True),
            artifact_uri=str(artifact_path),
            status=self.status,
        )
        self.session.add(registry)
        metrics = {
            "patterns_trained": [self.pattern_id],
            "training_rows": len(examples),
            "model_id": model_id,
            "artifact_uri": str(artifact_path),
            "feature_schema_hash": schema_hash,
            "cv_metrics": cv_metrics,
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
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
        },
    )
    print(json.dumps(result.metrics, sort_keys=True, default=str))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

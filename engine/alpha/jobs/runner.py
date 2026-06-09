"""
Evidence-backed job runner.

Wraps any BaseJob in evidence_jobs / evidence_job_runs bookkeeping:
  - Creates or finds the evidence_jobs row
  - Starts evidence_job_runs
  - Executes job.run(ctx)
  - Records finish with metrics/hashes/errors
  - Guarantees failed jobs are recorded as failed, never swallowed
"""

from __future__ import annotations

import json
import traceback
from typing import Optional

from sqlalchemy.orm import Session

from alpha.db.models import EvidenceJob, EvidenceJobRun
from alpha.evidence.writer import create_job, finish_run, start_run
from alpha.jobs.contracts import BaseJob, JobContext, JobResult


def run_job(
    session: Session,
    job: BaseJob,
    *,
    params: Optional[dict] = None,
    app_commit_sha: Optional[str] = None,
    dry_run: bool = False,
) -> JobResult:
    """
    Execute a job with full evidence bookkeeping.

    1. Find or create the evidence_jobs row.
    2. Start an evidence_job_runs row.
    3. Run the job.
    4. Record finish (or failure) on the run row.
    5. Commit.
    """
    # 1. Find or create job definition
    existing = (
        session.query(EvidenceJob)
        .filter(EvidenceJob.job_name == job.job_name)
        .first()
    )
    if existing:
        evidence_job = existing
    else:
        evidence_job = create_job(
            session,
            name=job.job_name,
            job_type=job.job_type,
            owner=job.owner_component,
        )

    # 2. Start run
    run = start_run(
        session,
        job_id=evidence_job.job_id,
        params=params,
        app_commit_sha=app_commit_sha,
    )
    session.flush()
    run_id = run.job_run_id
    session.commit()

    ctx = JobContext(
        job_id=evidence_job.job_id,
        job_run_id=run_id,
        started_at=run.started_at,
        params=params or {},
        app_commit_sha=app_commit_sha,
        dry_run=dry_run,
    )

    # 3. Execute — catch everything so failures are always recorded
    try:
        result = job.run(ctx)
    except KeyboardInterrupt as exc:
        session.rollback()
        result = JobResult(
            status="failed",
            errors=[{"exception": str(exc), "traceback": traceback.format_exc()}],
        )
    except Exception as exc:
        session.rollback()
        result = JobResult(
            status="failed",
            errors=[{"exception": str(exc), "traceback": traceback.format_exc()}],
        )

    # 4. Record finish
    run = session.get(EvidenceJobRun, run_id)
    if run is None:
        raise RuntimeError(f"Evidence job run disappeared before finish: {run_id}")
    finish_run(
        session,
        run,
        status=result.status,
        metrics=result.metrics or None,
        input_hashes=result.input_hashes or None,
        output_hashes=result.output_hashes or None,
        error={"errors": result.errors} if result.errors else None,
    )

    # 5. Commit
    session.commit()
    return result

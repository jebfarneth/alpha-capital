"""
Job runner contracts.

Every evidence-producing process is a job. Jobs implement BaseJob.run()
and return a JobResult. The runner wraps execution in evidence_job_runs
bookkeeping so failures are always recorded.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class JobContext:
    """Execution context passed to every job."""

    job_id: str
    job_run_id: str
    started_at: datetime
    params: Dict[str, Any] = field(default_factory=dict)
    app_commit_sha: Optional[str] = None
    dry_run: bool = False


@dataclass
class JobResult:
    """Outcome returned by every job."""

    status: str  # finished, failed
    metrics: Dict[str, Any] = field(default_factory=dict)
    input_hashes: Dict[str, str] = field(default_factory=dict)
    output_hashes: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True when the job completed with finished status."""

        return self.status == "finished"


class BaseJob(abc.ABC):
    """Protocol for evidence-backed jobs."""

    @property
    @abc.abstractmethod
    def job_name(self) -> str:
        """Stable job name persisted in evidence job metadata."""

        ...

    @property
    @abc.abstractmethod
    def job_type(self) -> str:
        """High-level job category persisted for operations and audits."""

        ...

    @property
    def owner_component(self) -> str:
        """Component owner recorded on evidence job rows."""

        return "alpha_engine"

    @abc.abstractmethod
    def run(self, ctx: JobContext) -> JobResult:
        """Execute the job under the supplied evidence context."""

        ...

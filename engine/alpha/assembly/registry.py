"""
Feature assembly registry.

Tracks all 17 pattern ids and their assembly status. The registry is the
single authority for which patterns have production assemblers, which have
detectors only, and which are reserved/disabled future slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from alpha.patterns.contracts import PatternId, PatternInput


class AssemblerStatus:
    """Lifecycle states for pattern feature assemblers."""

    IMPLEMENTED = "implemented"
    DETECTOR_ONLY = "detector_only"
    RESERVED = "reserved"
    DISABLED = "disabled"


@dataclass
class AssemblerRegistryEntry:
    """Registry row describing one pattern's assembler availability."""

    pattern_id: str
    status: str
    assembler: Optional[Callable[..., Any]] = None


class AssemblyRegistry:
    """Registry mapping all 17 pattern ids to assembly status and assembler callables."""

    # Patterns with callable detectors in the codebase
    _DETECTOR_PATTERNS = {"M1", "M2", "M3", "M4", "M5", "M6", "M7", "I1", "I8"}

    def __init__(
        self,
        assemblers: Optional[Dict[str, Callable[..., Any]]] = None,
        disabled: Optional[set] = None,
    ):
        self._assemblers = assemblers or {}
        self._disabled = disabled or set()
        self._entries: Dict[str, AssemblerRegistryEntry] = {}
        self._build()

    def _build(self) -> None:
        for pid in PatternId.ALL:
            if pid in self._disabled:
                status = AssemblerStatus.DISABLED
                assembler = None
            elif pid in self._assemblers:
                status = AssemblerStatus.IMPLEMENTED
                assembler = self._assemblers[pid]
            elif pid in self._DETECTOR_PATTERNS:
                status = AssemblerStatus.DETECTOR_ONLY
                assembler = None
            else:
                status = AssemblerStatus.RESERVED
                assembler = None
            self._entries[pid] = AssemblerRegistryEntry(
                pattern_id=pid, status=status, assembler=assembler,
            )

    def get(self, pattern_id: str) -> AssemblerRegistryEntry:
        """Return the registry entry for a known pattern id."""

        entry = self._entries.get(pattern_id)
        if entry is None:
            raise KeyError(f"unknown pattern_id: {pattern_id}")
        return entry

    def status(self, pattern_id: str) -> str:
        """Return only the assembler status for a known pattern id."""

        return self.get(pattern_id).status

    def all_entries(self) -> List[AssemblerRegistryEntry]:
        """Return all pattern entries in registry order."""

        return list(self._entries.values())

    def implemented_ids(self) -> List[str]:
        """Return pattern ids with production assemblers available."""

        return [e.pattern_id for e in self._entries.values()
                if e.status == AssemblerStatus.IMPLEMENTED]

    def diagnostics(self) -> Dict[str, str]:
        """Return a status map for all 17 patterns."""
        return {e.pattern_id: e.status for e in self._entries.values()}

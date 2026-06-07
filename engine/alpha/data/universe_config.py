"""Shared operating-universe bounds.

These constants define the canonical global PIT operating-universe base.
Individual patterns may apply validation or execution overlays, but those
overlays must not be confused with the canonical universe ceiling.
"""

MCAP_MIN = 30_000_000
MCAP_MAX = 5_000_000_000

MARKET_CAP_BUCKETS = (
    ("30m_100m", 30_000_000, 100_000_000),
    ("100m_250m", 100_000_000, 250_000_000),
    ("250m_500m", 250_000_000, 500_000_000),
    ("500m_1b", 500_000_000, 1_000_000_000),
    ("1b_2b", 1_000_000_000, 2_000_000_000),
    ("2b_5b", 2_000_000_000, 5_000_000_000),
)
MARKET_CAP_BUCKET_LABELS = tuple(label for label, _, _ in MARKET_CAP_BUCKETS)

# Default FMP fetch grid for the global base. This keeps the dense microcap
# region granular while avoiding hundreds of uniform $10M slices up to $5B.
DEFAULT_SLICE_GRID = (
    (30_000_000, 250_000_000, 10_000_000),
    (250_000_000, 1_000_000_000, 50_000_000),
    (1_000_000_000, 2_000_000_000, 100_000_000),
    (2_000_000_000, 5_000_000_000, 250_000_000),
)

# Retained for explicit uniform-slice tests/operator overrides. It is not the
# default grid for the widened global base universe.
SLICE_WIDTH = 10_000_000
MIN_SLICE_WIDTH = 1_000_000

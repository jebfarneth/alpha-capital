"""Versioned Polygon SIC-to-sector bucketing for M3.

The production M3 taxonomy is Polygon as-of SIC, bucketed by 2-digit
major group into a GICS-like sector count. Keep this map versioned so
future taxonomy-sensitivity tests can compare production and challenger
partitions without silently mutating historical meaning.
"""

from __future__ import annotations

from typing import Dict, Optional


SIC_TO_SECTOR_MAP_VERSION = "POLYGON_SIC_2DIGIT_V1_2026_06_05"

SECTORS = (
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
)


_MAJOR_GROUP_TO_SECTOR: Dict[int, str] = {
    # Agriculture, forestry, fishing.
    **{major: "Consumer Staples" for major in range(1, 10)},
    # Mining, oil/gas extraction, petroleum refining.
    **{major: "Energy" for major in (10, 12, 13, 14, 29)},
    # Construction and broad industrial production/distribution.
    **{major: "Industrials" for major in (15, 16, 17, 24, 25, 32, 33, 34)},
    # Food, tobacco, grocery.
    **{major: "Consumer Staples" for major in (20, 21, 54)},
    # Textiles, apparel, retail, lodging, leisure, personal services.
    **{major: "Consumer Discretionary" for major in (
        22, 23, 31, 39, 52, 53, 55, 56, 57, 58, 59, 70, 72, 75, 76, 78, 79, 82,
    )},
    # Paper, printing, rubber/plastics, primary materials.
    **{major: "Materials" for major in (26, 27, 30)},
    # Pharma/chemicals, instruments, health services.
    **{major: "Health Care" for major in (28, 38, 80, 83, 84)},
    # Machinery, electronics, software/data services.
    **{major: "Information Technology" for major in (35, 36, 73)},
    # Transportation, wholesale, professional and repair services.
    **{major: "Industrials" for major in (
        37, 40, 41, 42, 44, 45, 46, 47, 50, 51, 74, 81, 86, 87, 89,
    )},
    # Telecommunications and media.
    48: "Communication Services",
    # Electric/gas/sanitary utilities.
    49: "Utilities",
    # Finance/insurance and real-estate split.
    **{major: "Financials" for major in (60, 61, 62, 63, 64, 67)},
    65: "Real Estate",
}


def normalize_sic_code(sic_code: object) -> Optional[str]:
    """Return a zero-padded 4-digit SIC code when parseable."""

    if sic_code is None:
        return None
    text = str(sic_code).strip()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    if len(digits) > 4:
        digits = digits[:4]
    return digits.zfill(4)


def sector_for_sic(sic_code: object) -> Optional[str]:
    """Map a raw SIC code to the production M3 sector bucket."""

    normalized = normalize_sic_code(sic_code)
    if normalized is None:
        return None
    major_group = int(normalized[:2])
    return _MAJOR_GROUP_TO_SECTOR.get(major_group)


def major_group_map() -> Dict[int, str]:
    """Return a copy of the versioned major-group map for tests/audits."""

    return dict(_MAJOR_GROUP_TO_SECTOR)

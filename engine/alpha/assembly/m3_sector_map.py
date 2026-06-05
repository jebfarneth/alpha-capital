"""Versioned Polygon SIC-to-sector bucketing for M3.

The production M3 taxonomy is Polygon as-of SIC, bucketed into a GICS-like
sector count. A 2-digit major-group map is the fallback, but several SIC
major groups are economically mixed enough to need checked-in 3-digit
overrides. Keep this map versioned so future taxonomy-sensitivity tests can
compare production and challenger partitions without silently mutating
historical meaning.
"""

from __future__ import annotations

from typing import Dict, Optional


SIC_TO_SECTOR_MAP_VERSION = "POLYGON_SIC_PREFIX_V2_2026_06_05"

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
    # Chemicals are materials by default; drug and biotech SICs override below.
    28: "Materials",
    # Measuring/medical instruments are mixed; medical instruments override below.
    38: "Industrials",
    # Health services and social services.
    **{major: "Health Care" for major in (80, 83, 84)},
    # Industrial machinery by default; computer equipment overrides below.
    35: "Industrials",
    # Electronics, semiconductors, software/data services.
    **{major: "Information Technology" for major in (36, 73)},
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


_SIC_PREFIX_OVERRIDES: Dict[str, str] = {
    # 28 is broad chemicals; drugs/biotech are health care.
    "283": "Health Care",
    # 35 is broad machinery; computer/storage equipment is technology.
    "357": "Information Technology",
    # 38 is broad instruments; surgical/medical instruments and ophthalmic goods
    # are health care rather than industrial instrumentation.
    "384": "Health Care",
    "385": "Health Care",
    # 73 is broad business services; ads and miscellaneous services are not IT.
    "731": "Communication Services",
    "738": "Industrials",
}


_FMP_SECTOR_TO_CANONICAL: Dict[str, str] = {
    "basic materials": "Materials",
    "communication services": "Communication Services",
    "consumer cyclical": "Consumer Discretionary",
    "consumer defensive": "Consumer Staples",
    "consumer discretionary": "Consumer Discretionary",
    "consumer staples": "Consumer Staples",
    "energy": "Energy",
    "financial services": "Financials",
    "financials": "Financials",
    "health care": "Health Care",
    "healthcare": "Health Care",
    "industrials": "Industrials",
    "real estate": "Real Estate",
    "technology": "Information Technology",
    "information technology": "Information Technology",
    "utilities": "Utilities",
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
    for width in (4, 3):
        override = _SIC_PREFIX_OVERRIDES.get(normalized[:width])
        if override is not None:
            return override
    major_group = int(normalized[:2])
    return _MAJOR_GROUP_TO_SECTOR.get(major_group)


def canonical_sector_from_fmp(value: object) -> Optional[str]:
    """Normalize an FMP sector label to the canonical M3 11-sector vocabulary."""

    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    return _FMP_SECTOR_TO_CANONICAL.get(text.casefold())


def major_group_map() -> Dict[int, str]:
    """Return a copy of the versioned major-group map for tests/audits."""

    return dict(_MAJOR_GROUP_TO_SECTOR)

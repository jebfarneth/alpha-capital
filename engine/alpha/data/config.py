"""
Adapter configuration from environment variables.

Never stores secrets in code. Reads os.environ only.
Raises ConfigError with a clear message when a required key is missing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


class ConfigError(Exception):
    """Raised when a required environment variable is missing."""

    pass


def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise ConfigError(
            f"Required environment variable {key!r} is not set. "
            f"See engine/.env.example for the full list."
        )
    return val


def _optional(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class FmpConfig:
    api_key: str
    base_url: str = "https://financialmodelingprep.com"

    @classmethod
    def from_env(cls) -> FmpConfig:
        return cls(
            api_key=_require("FMP_API_KEY"),
            base_url=_optional("FMP_BASE_URL", cls.base_url),
        )


@dataclass(frozen=True)
class AlpacaConfig:
    api_key: str
    secret_key: str
    base_url: str  # paper or live

    @classmethod
    def from_env(cls) -> AlpacaConfig:
        return cls(
            api_key=_require("ALPACA_API_KEY"),
            secret_key=_require("ALPACA_SECRET_KEY"),
            base_url=_optional(
                "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
            ),
        )


@dataclass(frozen=True)
class PolygonConfig:
    api_key: str
    base_url: str = "https://api.polygon.io"

    @classmethod
    def from_env(cls) -> PolygonConfig:
        return cls(
            api_key=_require("POLYGON_API_KEY"),
            base_url=_optional("POLYGON_BASE_URL", cls.base_url),
        )

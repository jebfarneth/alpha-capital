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
    """Configuration for the Financial Modeling Prep adapter."""

    api_key: str
    base_url: str = "https://financialmodelingprep.com"

    @classmethod
    def from_env(cls) -> FmpConfig:
        """Build FMP configuration from process environment variables."""

        return cls(
            api_key=_require("FMP_API_KEY"),
            base_url=_optional("FMP_BASE_URL", cls.base_url),
        )


@dataclass(frozen=True)
class AlpacaConfig:
    """Configuration for Alpaca broker and market-data adapters."""

    api_key: str
    secret_key: str
    base_url: str  # paper or live

    @classmethod
    def from_env(cls) -> AlpacaConfig:
        """Build Alpaca configuration from process environment variables."""

        return cls(
            api_key=_require("ALPACA_API_KEY"),
            secret_key=_require("ALPACA_SECRET_KEY"),
            base_url=_optional(
                "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
            ),
        )


@dataclass(frozen=True)
class PolygonConfig:
    """Configuration for Polygon/Massive reference-data adapters."""

    api_key: str
    base_url: str = "https://api.polygon.io"

    @classmethod
    def from_env(cls) -> PolygonConfig:
        """Build Polygon configuration from process environment variables."""

        return cls(
            api_key=_require("POLYGON_API_KEY"),
            base_url=_optional("POLYGON_BASE_URL", cls.base_url),
        )


@dataclass(frozen=True)
class BenzingaConfig:
    """Configuration for Benzinga event-data adapters."""

    api_key: str
    base_url: str = "https://api.benzinga.com"

    @classmethod
    def from_env(cls) -> BenzingaConfig:
        """Build Benzinga configuration from process environment variables."""

        api_key = os.environ.get("BENZINGA_API_KEY") or os.environ.get("BENZINGA_TOKEN")
        if not api_key:
            raise ConfigError(
                "Required environment variable 'BENZINGA_API_KEY' or "
                "'BENZINGA_TOKEN' is not set. See engine/.env.example for the full list."
            )
        return cls(
            api_key=api_key,
            base_url=_optional("BENZINGA_BASE_URL", cls.base_url),
        )

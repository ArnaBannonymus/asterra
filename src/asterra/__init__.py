"""Asterra: Support-aware machine learning for Earth observation."""

from __future__ import annotations

__all__ = [
    "__version__",
    "BandSchema",
    "EOData",
    "SupportSpec",
    "SupportMatrix",
]

__version__ = "0.1.0"

from .data.bands import BandSchema
from .data.eodata import EOData
from .data.support import SupportSpec
from .support.matrix import SupportMatrix


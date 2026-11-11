"""Jordan Currency Detector: detect, count and value Jordanian coins and banknotes."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jordan-currency-detector")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]

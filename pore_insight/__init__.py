import importlib.metadata
import logging

try:
    __version__ = importlib.metadata.version("pore_insight")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"  # Fallback if the package is not installed

# Package-level logger. By default no handler is attached, so nothing is printed.
# Users can enable output with:
#   logging.basicConfig(level=logging.INFO)
# or silence specific levels with:
#   logging.getLogger('pore_insight').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
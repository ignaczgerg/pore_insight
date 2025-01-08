import importlib.metadata

try:
    __version__ = importlib.metadata.version("pore_insight")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"  # Fallback if the package is not installed
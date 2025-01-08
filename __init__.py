import importlib.metadata

try:
    __version__ = importlib.metadata.version("PoreInsight")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"  # Fallback if the package is not installed
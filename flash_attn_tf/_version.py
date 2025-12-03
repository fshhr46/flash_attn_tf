__all__ = ["__version__"]

try:
    from importlib.metadata import version as _get_version

    # derive version from package metadata
    __version__ = _get_version("flash-attn-tf")
except Exception:
    __version__ = "0.1.0"

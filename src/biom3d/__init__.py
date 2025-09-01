from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("biom3d")
except PackageNotFoundError:
    __version__ = None 
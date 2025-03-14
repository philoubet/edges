"""
edges: A Python package for calculating the environmental impact of products by
applying characterization factors conditioned by the context of exchanges.
"""

__all__ = ("EdgeLCIA",)

__version__ = "0.1.0"

from bw2calc import __version__ as bw2calc_version

if isinstance(bw2calc_version, str):
    bw2calc_version = tuple(map(int, bw2calc_version.split(".")))

if bw2calc_version < (2, 0, 0):
    from .edgelcia_bw2 import EdgeLCIA

else:
    from .edgelcia import EdgeLCIA
from .utils import get_available_methods

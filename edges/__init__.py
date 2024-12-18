"""
edges: A Python package for calculating the environmental impact of products by
applying characterization factors conditioned by the context of exchanges.
"""

__all__ = ("EdgeLCIA",)

__version__ = "0.1.0"

from .edgelcia import EdgeLCIA
from .utils import get_available_methods

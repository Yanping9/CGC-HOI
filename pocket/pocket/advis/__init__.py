"""
The advanced visual package




"""

try:
    import seaborn
except ImportError:
    raise ImportError(
        "pocket.advis requires the package seaborn. "
        "Please run pip install seaborn."
    )

from .colours import *
from .heatmap import *
from .text import *
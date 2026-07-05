import importlib
import sys

from .config import P_DEMO, P_MIMIC, P_ROOT, hello
from .data import load_data


def reload(module=None):
    """Reload module (for use in Jupyter Notebook)

    Args:
        module (types.ModuleType, optional): module to reload
    """
    importlib.reload(module or sys.modules[__name__])


__all__ = [
    "P_ROOT",
    "P_MIMIC",
    "P_DEMO",
    "hello",
    "reload",
    "load_data",
]

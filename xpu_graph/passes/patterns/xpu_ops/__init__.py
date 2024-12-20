import pkgutil
import importlib

from xpu_graph.passes.patterns.pattern import Pattern, AutoMatchPattern
from xpu_graph.config import OptLevel
from xpu_graph.utils import logger

def get_all_patterns(opt_level: OptLevel):
    patterns = []

    for _, module_name, _ in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{module_name}")

        for name in dir(module):
            pat = getattr(module, name)
            if isinstance(pat, type) and \
                issubclass(pat, Pattern) and \
                pat not in (Pattern, AutoMatchPattern) and\
                pat._opt_level <= opt_level:

                patterns.append(pat())

    logger.debug(f"xpu_graph enable builtin xpu_ops patterns: {[pat.__class__.__name__ for pat in patterns]}")

    return patterns
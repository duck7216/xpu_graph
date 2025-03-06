from .compiler import XpuGraph
from .config import Target, OptLevel, XpuGraphConfig
from .cache import XpuGraphCache, default_cache
from typing import Dict, Any

__all__ = ["XpuGraph", "XpuGraphConfig", "Target", "OptLevel"]


def mlu_compiler(
    freeze: bool = True,
    opt_level: OptLevel = OptLevel.level1,
    constant_folding: bool = True,
    cache: XpuGraphCache = default_cache(),
    debug: bool = False,
    vendor_compiler_config: Dict[str, Any] = {"mode": "reduce-overhead"},
):
    config = XpuGraphConfig(
        target=Target.mlu,
        freeze=freeze,
        opt_level=opt_level,
        constant_folding=constant_folding,
        debug=debug,
        vendor_compiler_config=vendor_compiler_config,
    )
    return XpuGraph(config, cache)

def npu_compiler(
    freeze: bool = False,
    opt_level: OptLevel = OptLevel.level1,
    graph: bool = True,
    constant_folding: bool = False,
    cache: XpuGraphCache = default_cache(),
):
    config = XpuGraphConfig(
        target=Target.npu,
        freeze=freeze,
        opt_level=opt_level,
        constant_folding=constant_folding,
    )
    if graph:
        config.vendor_compiler = {"mode": "default"}
    return XpuGraph(config, cache)
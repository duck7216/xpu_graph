import math
import pytest

import torch
import xpu_graph

from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)


def fn0(input, input_num, bs, in_dim):
    sqa = torch.arange(input_num, device="mlu:0").reshape(1, -1)
    sqa = sqa.expand(bs, -1)
    sqa = sqa.unsqueeze(-1).repeat(1, 1, in_dim)
    return input.gather(index=sqa, dim=1)


def gather_test(xpu_graph_backend, func):
    batch = 86
    in_dim = 256
    dtype = torch.half
    input = torch.randn(batch, 496, in_dim, dtype=dtype, device="mlu")
    res1 = func(input, 46, batch, in_dim)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res = compiled(input, 46, batch, in_dim)

    assertTensorsEqual(
        res.cpu().float(), res1.cpu().float(), 0.005, use_MSE=True, use_RAE=True
    )


class TestGather:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False, freeze=True, opt_level=OptLevel.level2
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0],
    )
    def test_gather_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            gather_test(self.xpu_graph_backend, pattern_func)


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False, freeze=True, opt_level=OptLevel.level1
    )
    gather_test(xpu_graph_backend, fn0)

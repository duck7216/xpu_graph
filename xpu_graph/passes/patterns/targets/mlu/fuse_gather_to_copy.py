from typing import Optional, Tuple, Union

import torch
from torch import nn, fx
import torch_mlu
import torch_mlu_ops
from xpu_graph.utils import logger
from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from ...utils.check_ops import (
    get_shape,
)

TensorShape = Union[torch.Size, Tuple[int, ...]]
NodeType = fx.Node


class FusedSliceReplacement(nn.Module):
    def forward(self, input, slice_dim, slice_start, slice_end):
        return input.narrow(
            slice_dim, slice_start, slice_end - slice_start
        ).contiguous()


"""
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arange : [num_users=1] = call_function[target=torch.ops.aten.arange.default](args = (46,), )
    %view : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%arange, [1, -1]), kwargs = {})
    %expand : [num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view, [86, -1]), kwargs = {})
    %unsqueeze : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%expand, -1), kwargs = {})
    %repeat : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%unsqueeze, [1, 1, 256]), kwargs = {})
    %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%arg0_1, 1, %repeat), kwargs = {})
"""


class FusedGatherToCopy(Pattern):
    _opt_level = OptLevel.level1

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        graph_module.add_submodule("mlu_gather_to_copy", FusedSliceReplacement())
        candidates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten.gather.default
        ]
        for gather_node in candidates:
            repeat_node = gather_node.args[2]
            gather_dim = gather_node.args[1]
            if repeat_node.target != torch.ops.aten.repeat.default:
                continue
            unsqueeze_node = repeat_node.args[0]
            if unsqueeze_node.target != torch.ops.aten.unsqueeze.default:
                continue
            expand_node = unsqueeze_node.args[0]
            if expand_node.target != torch.ops.aten.expand.default:
                continue
            view_node = expand_node.args[0]
            if view_node.target != torch.ops.aten.view.default:
                continue
            arange_node = view_node.args[0]
            if arange_node.target != torch.ops.aten.arange.default:
                continue

            slice_start = 0
            slice_end = 0
            if len(arange_node.args) == 1:
                slice_end = arange_node.args[0]
            else:
                # (TODO JYJ) add more pattern
                continue

            if view_node.args[1] != [1, -1]:
                continue
            arange_dim = -1
            # (TODO JYJ) add more pattern
            if unsqueeze_node.args[1] != -1:
                continue
            arange_dim = -2
            repeat_node_shape = get_shape(repeat_node)
            arange_dim = len(repeat_node_shape) + arange_dim
            if arange_dim != gather_dim:
                continue
            changed = True

            with graph_module.graph.inserting_before(gather_node):
                new_node = graph_module.graph.call_module(
                    "mlu_gather_to_copy",
                    args=(gather_node.args[0], arange_dim, slice_start, slice_end),
                )
            gather_node.replace_all_uses_with(new_node)
            graph_module.graph.erase_node(gather_node)
            graph_module.graph.lint()
            graph_module.recompile()

        return is_modified

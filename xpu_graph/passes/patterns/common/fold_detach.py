import torch
import torch.fx as fx
from xpu_graph.passes.patterns.pattern import Pattern


class FoldDetach(Pattern):

    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten.detach.default
        ]
        for detach in candidates:
            inp = detach.args[0]
            changed = True
            detach.replace_all_uses_with(inp)
            gm.graph.erase_node(detach)

        gm.graph.lint()
        gm.recompile()
        return changed

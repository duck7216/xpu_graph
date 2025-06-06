import torch.fx as fx

from xpu_graph.passes.optimizer import Optimizer

class Dce(Optimizer):
    def process(self, gm: fx.GraphModule):
        # We delete 'get_attr' manually, because we need to remove the constant also.
        get_attr_nodes = [node for node in gm.graph.nodes if node.op == "get_attr"]
        for node in get_attr_nodes:
            if len(node.users) != 0:
                continue
            if hasattr(gm, node.target):
                delattr(gm, node.target)
            gm.graph.erase_node(node)

        return gm.graph.eliminate_dead_code()
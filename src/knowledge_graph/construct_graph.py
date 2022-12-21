"""
Created on Fri Mar 26 2021

@author Name Redacted Surname Redacted
"""


from networkx import MultiDiGraph
from src.knowledge_graph.construct_tree import construct_tree
from tqdm import tqdm
from src.knowledge_graph.applicable_actions import ActionsSupplier


def constrained_construct_tree(seed, history, initializer, supplier=ActionsSupplier):
    """Construct a tree from seed, expanding only in case the knowledge will be enriched.
    To this extent, the applicable_actions method of the supplier is overridden, and a filtering is applied based on what we know yet.

    Idea: If the node `n` is known, and given feature `f`, if `f(n)` is known, then I can avoid doing `f(n)` again, we won't discover anything new.

    Notice: the behaviour of the initializer is not controlled by this method.

    Args:
        seed (dict): Tree seed
        history (set): Tuples (feature, node) applied so far;
        initializer (callable): Function to be called for initializing the tree from seed. required by construct_tree;
        supplier (class, optional): Class that supplies actions to apply to construct_tree iteratively. Defaults to ActionsSupplier.

    Returns:
        nx DiGraph: The tree.
        history: The history, updated.
    """

    class ConstrainedActionsSupplier(supplier):

        def applicable_actions(self):
            actions = super(ConstrainedActionsSupplier, self).applicable_actions()
            constrained_actions = []
            for action in actions:
                signature = (action[0].__name__, self.t.nodes()[action[1]]['mergiable_id'])
                if signature not in history:
                    constrained_actions.append(action)
                    history.add(signature)
            return constrained_actions

    tree = construct_tree(seed, supplier=ConstrainedActionsSupplier(), initializer=initializer)
    return tree, history


def compose(graph, tree):
    """Merge tree in the knowledge graph.

    Assumptions:
    - mergiable_id field should exist for tree root i.e. `source`.

    Args:
        graph (nx graph)
        tree (nx graph)

    Returns:
        graph: the graph resulting from the merge.
    """
    def recurse(n1, n2):
        graph.add_node(tree.nodes()[n2]['mergiable_id'])
        graph.add_edge(tree.nodes()[n1]['mergiable_id'], tree.nodes()[n2]['mergiable_id'], type=tree[n1][n2]['type'])
        [recurse(n2, n) for n in tree[n2]]

    assert tree.nodes()['source']['mergiable_id'] not in graph
    graph.add_node(tree.nodes()['source']['mergiable_id'])
    [recurse('source', n) for n in tree['source']]
    return graph


def construct_graph(seeds, initializer, supplier=ActionsSupplier):
    """Construct a knowledge graph.

    It works in two steps iteratively:
    1) Build a tree from a seed;
    2) Merge the tree in the graph.

    Notice: it is smart, it does not compute parts of the tree which are in the graph yet.

    The graph has:
    - mergiable_id of tree nodes, as node ids;
    - type of tree edges, as edge types.

    Args:
        seeds (list): List of seeds to pass to construct_tree;
        initializer (callable): initilizer to use in construct_tree.
                                It should provide a mergiable_id also for source node;
        supplier (object, optional): Action supplier to use in construct_tree;

    Returns:
        nx graph: The graph.
    """
    graph = MultiDiGraph()
    history = set()
    for seed in tqdm(seeds):
        tree, history = constrained_construct_tree(seed, history, initializer, supplier)
        graph = compose(graph, tree)
    return graph

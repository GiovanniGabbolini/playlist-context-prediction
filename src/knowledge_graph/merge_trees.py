import networkx as nx
from tqdm import tqdm
from src.utils.utils_ngx_tree import tree_id


def merge_trees(generators, strategy_fields_source_node=lambda t: {}, strategy_tree_id=tree_id):
    """Given many trees, merge them in a unique knowldge graph.
    The unique graph is a multigraph.

    Notice: it assumes the root of the trees have id equal to 'source'.

    Args:
        generators (list): list of functions. if called, they return lists of trees
        strategy_fields_source_node (func): returns a dictionary to add to source nodes of trees in the merged graph
        strategy_tree_id (func): given a tree returns the identifier of the tree to be used in the graph.
                                 it will be possible to access the source node of the tree in the graph given than id.

    Returns:
        nx graph: merged knowledge graph
    """
    graph = nx.MultiDiGraph()

    print('Merging trees ...')
    for generator in tqdm(generators):
        for t in generator():

            id_tree = strategy_tree_id(t)
            assert id_tree not in graph

            graph.add_node(id_tree, type='source', value=id_tree, id=id_tree, **strategy_fields_source_node(t))

            # Element in queue represent nodes in the shape (id_node_in_tree, id_node_in_graph)
            q = [('source', id_tree)]
            while True:

                if len(q) == 0:
                    break
                else:
                    node = q.pop(0)

                    edges = t.edges(node[0])
                    for edge in edges:

                        n = t.nodes()[edge[1]].copy()

                        old_id = n['id']
                        new_id = n['mergiable_id']

                        n.pop('id')
                        n.pop('tree')
                        graph.add_node(new_id, id=new_id, **n)

                        type_edge = t[node[0]][old_id]['type']
                        graph.add_edge(node[1], new_id, type=type_edge)

                        q.append((old_id, new_id))

    return graph

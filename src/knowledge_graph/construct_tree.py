import networkx as nx
from src.text_processing.preprocess_music_seed_key import preprocess_music_seed_key
from src.knowledge_graph.applicable_actions import ActionsSupplier
import json


def initializer(seed):
    """Implements the default behaviour of the initializer as described in construct_tree.

    It assumes the music domain, and therefore it expects to have album_name, artist_name and track_name as keys in seed.
    Some of such fields might be missing, it will continue working.

    It preprocesses album_name and track_name, by removing everything between paranthesis and after '-'.

    Args:
        seed (dict)
    """
    t = nx.DiGraph()

    # Source dummy node
    # The value field of source node contains the id of the tree
    t.add_node('source', type='source', value='source', id='source', tree=t)

    if 'track_name' in seed:
        seed['track_name'] = preprocess_music_seed_key(seed['track_name'])
    if 'album_name' in seed:
        seed['album_name'] = preprocess_music_seed_key(seed['album_name'])

    # Initialize the tree with the attributes contained in the seed dictionary.
    for k in seed.keys():
        t.add_node(k, type=k, value=seed[k], id=k, tree=t)
        t.nodes()[k]['mergiable_id'] = craft_mergiable_id(t.nodes()[k])
        t.add_edge('source', k, type='init', generating_function='init')

    # Add also the album_name_artist_name node, if possible.
    # This node is required by a feature.
    if 'album_name' in seed and 'artist_name' in seed:
        k = 'album_name_artist_name'
        t.add_node(k, type=k, value={'album_name': seed['album_name'], 'artist_name': seed['artist_name']}, id=k, tree=t)
        t.nodes()[k]['mergiable_id'] = craft_mergiable_id(t.nodes()[k])
        t.add_edge('source', k, type='init', generating_function='init')

    return t


def construct_tree(seed, supplier=None, initializer=initializer):
    """Builds a tree from the seed.

       The tree construction is articulated in two parts:
       1) From seed, use initializer to create an initial tree t;
       2) Add nodes and edges to t by applying features provided by the supplier;

       **1)**
       The creation of t is handled by the initializer from the seed dictionary.
       The seed can be, for example: {
           'track_name': 'untitled1',
           'artist_name': 'aphex twin',
           'album_name': 'selected ambient works'
       }
       One way the initializer works is by creating five nodes:
       - source: there by default
       - track_name: of type track_name and value untitled1
       - artist_name: of type artist_name and value aphex twin
       - album_name: of type album_name and value selected ambient works
       - album_name_artist_name: ..
       And this is indeed how the initializer works by default.

       **2)**
       The tree is grown by applying the features. What features to apply to what nodes is told by the supplier.
       Every feature yield to the creation to new nodes and new edges, and the process repeats.

       Every feature returns a dictionary, which has a fixed structure:
       - value: value returned by the feature.
       - node_type (optional): type of the node from the feature.
       - edge_type (optional): type of the edge from the feature.
       If node_type and edge_type are missing, they are inferred from feature annotations.

       Every node in the tree shares a common internal structure:
       - id: Unique string identifier in the tree. Constructed concatenating the functions applied to
             obtain that node to the name of the node from which that node comes from
       - type: Node type
       - value: Node value
       - mergiable_id: Unique string identifier in the wild. Constructed by converting type and value to string, and concatenating them.
                       In practice, if two node have the same mergiable_id, then they will have same value and type, and therefore they'll be equal.
       - tree: reference to whole tree.

       Every edge in the tree share a common internal structure:
       - type: Edge type
       - generating_function: Name of the feature that generated the node

    Arguments:
        seed {dict} --
        supplier {obj} -- Supplies which actions (features) can be applied to the tree nodes;
        initializer {obj} -- Initialize an initial tree from the seed.

    Returns:
        t {nx graph} --
    """
    t = initializer(seed)

    action_supplier = ActionsSupplier() if supplier is None else supplier
    action_supplier.set_tree(t)

    while True:

        # Retrieve the applicable features
        actions = action_supplier.applicable_actions()

        if len(actions) == 0:
            break

        for action in actions:

            # Construct the actual dictionary to be passed to the feature
            feature = action[0]
            node = action[1]

            # Call the feature
            return_value = feature(t.nodes()[node])

            if return_value is None:
                continue

            assert type(return_value) == dict or type(return_value) == list
            return_value = [return_value] if type(return_value) == dict else return_value
            for idx, v in enumerate(return_value):

                # Resolve node type, edge type and node value
                edge_type = v.pop('edge_type') if 'edge_type' in v else feature.__name__
                node_type = v.pop('node_type') if 'node_type' in v else feature.__annotations__['return']
                value_node = v.pop('value')

                assert type(edge_type) == str and type(node_type) == str, "Nodes and edge types should be strings."
                assert len(v) == 0, f"Function {feature.__name__}: additional fields should be included in value."

                # Craft node id
                new_node = f"{node}~{feature.__name__}" if idx == 0 else f"{node}~{feature.__name__}-{idx}"

                generating_function = feature.__name__

                # Add
                assert new_node not in t and node in t
                t.add_node(new_node, value=value_node, type=node_type, id=new_node, tree=t)
                t.nodes()[new_node]['mergiable_id'] = craft_mergiable_id(t.nodes()[new_node])
                t.add_edge(node, new_node, type=edge_type, generating_function=generating_function)

    return t


def craft_mergiable_id(node):
    node_copy = node.copy()
    node_copy.pop('id')
    node_copy.pop('tree')
    s = ''
    keys = sorted(list(node_copy.keys()))
    for key in keys:
        s += f"~{key}:{node_copy[key] if type(node_copy[key])!=dict else json.dumps(node_copy[key], sort_keys=True)}"
    return s

from src.features.used import *
from src.features import used
from inspect import signature
from src.utils.utils_ngx_tree import father
from src.features.inspector import in_node_types, out_node_types, edge_types
from unittest.mock import MagicMock
import re


class ActionsSupplier:

    """Class thought as a supplier of features that allows to expand a tree.
       Therefore, handles the features that were already supplied (avoiding repetitions),
       considers recursive features, ..
    """

    def __init__(self):
        # This set contains the features already applied, in the form "node~feature",
        # which is a record of the features applied to nodes.
        self.applied_actions = set()

    def eligible_features(self):
        return used.__all__

    def set_tree(self, t):
        """Sets the tree from which the expansion begins.

        Args:
            t (nx graph)
        """
        self.t = t

        # If t was at least partially built, self.applied_actions should be set up correctly
        for node in t.nodes():
            idxs = [m.start() for m in re.finditer('~', node)]+[len(node)]
            for i, j in zip(idxs, idxs[1:]):
                self.applied_actions.add((node[:i], node[:j]))

    def eligible_nodes_filter(self, node, func):
        """Used to eventually filter out nodes that would otherwise be considered as applicable to some features.
        Useful to limit the tree growth.

        Active filters:
        -avoid infinite loops
        -do not expand nodes that originate from an artist relationship
        -if the feature is entailment, then apply to a synset only if it directly discend from a word
        -if the feature is member_holonyms, then apply to a synset only if it directly discend from a word
        -if the feature is member_meronyms, then apply to a synset only if it directly discend from a word
        -if the feature is part_holonyms, then apply to a synset only if it directly discend from a word
        -if the feature is part_meronyms, then apply to a synset only if it directly discend from a word
        -if the feature is substance_holonyms, then apply to a synset only if it directly discend from a word
        -if the feature is substance_meronyms, then apply to a synset only if it directly discend from a word
        -if the feature is synset_also_sees, then apply to a synset only if it directly discend from a word
        -if the feature is synset_attributes, then apply to a synset only if it directly discend from a word
        -if the feature is synset_similar_tos, then apply to a synset only if it directly discend from a word
        -if the feature is synset_verb_groups, then apply to a synset only if it directly discend from a word
        -if the feature is hypernyms, then apply to a synset only if it directly discend from a word
        -if the feature is hyponyms, then apply to a synset only if it directly discend from a word

        Args:
            node (nx Node): The eligible node to keep or filter out
            func (callable): Feature to which node would be eventually fed

        Returns:
            Bool: True if keep it, False if discard it
        """
        # a => b is equivalent to not a or b
        if father(node) is None or (
            self.t[father(node)['id']][node['id']]['generating_function'] != func.__name__ and
            self.t[father(node)['id']][node['id']]['generating_function'] != 'artist_relationships' and
            (not func.__name__ == 'entailment' or father(node)['type'] == 'word') and
            (not func.__name__ == 'member_holonyms' or father(node)['type'] == 'word') and
            (not func.__name__ == 'member_meronyms' or father(node)['type'] == 'word') and
            (not func.__name__ == 'part_holonyms' or father(node)['type'] == 'word') and
            (not func.__name__ == 'part_meronyms' or father(node)['type'] == 'word') and
            (not func.__name__ == 'substance_holonyms' or father(node)['type'] == 'word') and
            (not func.__name__ == 'substance_meronyms' or father(node)['type'] == 'word') and
            (not func.__name__ == 'synset_also_sees' or father(node)['type'] == 'word') and
            (not func.__name__ == 'synset_attributes' or father(node)['type'] == 'word') and
            (not func.__name__ == 'synset_similar_tos' or father(node)['type'] == 'word') and
            (not func.__name__ == 'synset_verb_groups' or father(node)['type'] == 'word') and
            (not func.__name__ == 'hypernyms' or father(node)['type'] == 'word') and
            (not func.__name__ == 'hyponyms' or father(node)['type'] == 'word')
        ):

            return True
        else:
            return False

    def applicable_actions(self):
        """Returns a list of actions i.e. features to be applied to a node in the tree.
        It draws from the elibible features, set by the eligible_features method.

        Returns:
            list -- List of tuples, [(feature, node), ... ].
        """
        l = []
        features = self.eligible_features()
        for f in features:
            func = getattr(globals()[f], f)
            l += self.applicable_actions_given_feature(func)
        return l

    def applicable_actions_given_feature(self, func):
        """Finds all the nodes in the tree that can be applied as the argument of the feature func.

        A node is applicable if the type matches the node types accepted by a feature.

        Arguments:
            func {callable} -- a feature

        Returns:
            list -- List of tuples, [(feature, node), ...].
        """
        sig = signature(func)

        assert len(sig.parameters.keys()) == 1, "A feature should have just one parameter."

        arg = next(iter(sig.parameters.keys()))
        # in_node_types tells the node types accepted by a feature.
        eligible_types_arg = in_node_types(func)[arg]
        possible_nodes = []

        for k in self.t.nodes():
            node = self.t.nodes()[k]
            if node['type'] in eligible_types_arg:

                if self.eligible_nodes_filter(node, func):
                    possible_nodes.append(k)

        # Filter combination based on if I have already applied that feature to that node
        applicable_nodes = []
        for node in possible_nodes:
            if f"{node}~{func.__name__}" not in self.applied_actions:
                applicable_nodes.append(node)

                # Update the record of applied actions.
                self.applied_actions.add(f"{node}~{func.__name__}")

        return [(func, node) for node in applicable_nodes]


class MockedActionsSupplier(ActionsSupplier):

    """Shares the logic with ActionSupplier, but returns mocked features instead of real features.
       Useful for testing and debugging, to construct a tree where no calls and feature computations are needed
    """

    def applicable_actions(self):
        actions = super(MockedActionsSupplier, self).applicable_actions()

        actions_mocked = []
        for action in actions:
            func = action[0]

            return_value = []

            e_types = edge_types(func)
            for edge_type in e_types:
                for node_type in out_node_types(func, edge_type=edge_type):
                    return_value.append({'value': 'mocked', 'node_type': node_type, 'edge_type': edge_type})

            mock = MagicMock(return_value=return_value if len(return_value) > 1 else return_value[0])
            mock.__annotations__ = func.__annotations__
            mock.__name__ = func.__name__

            actions_mocked.append((mock, action[1]))

        return actions_mocked


class CustomReturnValueMockedActionsSupplier(MockedActionsSupplier):

    """A MockedActionSupplier customizable in the mocked values returned by mocked features
       The dictionary d is shaped as follows:
       {
            'feature_name': return value,
            ...
       }
       It considers elibible actions only the keys of d 
       and exploits the logic of MockedActionsSupplier,
       but substitute mocked return values with values of d.
    """

    def __init__(self, d):
        super(CustomReturnValueMockedActionsSupplier, self).__init__()
        self.d = d

    def eligible_features(self):
        return list(self.d.keys())

    def applicable_actions(self):
        v = super(CustomReturnValueMockedActionsSupplier, self).applicable_actions()
        for e in v:
            e[0].return_value['value'] = self.d[e[0].__name__]
        return v


class InformativeActionSupplier(ActionsSupplier):

    """Discard all informative part of the tree.
    This is the same as discarding all nodes from token_phrase.
    """

    def eligible_features(self):
        return list(set(used.__all__)-set(['token_phrase']))

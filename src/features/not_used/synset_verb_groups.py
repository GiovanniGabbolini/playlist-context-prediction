"""
Created on --

@author Name Redacted Surname Redacted
"""

from nltk.corpus import wordnet as wn


def synset_verb_groups(synset) -> 'synset':
    """Returns the verb group synsets of a given synset, according to WordNet"""
    return [{'value': s.name()} for s in wn.synset(synset['value']).verb_groups()]

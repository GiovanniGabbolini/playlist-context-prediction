"""
Created on --

@author Name Redacted Surname Redacted
"""

from nltk.corpus import wordnet as wn

def hypernyms(synset) -> 'synset':
    """Returns hypernym synsets of a given synset, according to WordNet"""
    return [{'value': l.name()} for l in wn.synset(synset['value']).hypernyms()]

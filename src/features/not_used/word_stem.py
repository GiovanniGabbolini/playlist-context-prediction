"""
Created on --

@author Name Redacted Surname Redacted
"""
from src.text_processing.preprocess_word import stem


def word_stem(word) -> 'stem':
    """Returns the stem of a word"""
    return {'value': stem(word['value'])}

"""
Created on Tue Mar 31 2020

@author Name Redacted Surname Redacted
"""


def same_word_different_sense_phrase(n1, n2):
    """True if the two phrases share a word that assumes different meaning in those two phrases
    """

    if n1['value'] == n2['value']:

        if n1['meaning'] is not None and n2['meaning'] is not None and n1['meaning'] != n2['meaning']:

            return {'outcome': True,
                    'word': n1['value']}

    return {'outcome': False}

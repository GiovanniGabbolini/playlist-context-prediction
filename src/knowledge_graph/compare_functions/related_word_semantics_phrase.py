"""
Created on Tue Mar 31 2020

@author Name Redacted Surname Redacted
"""

# from src.data import data
import wordfreq
import math

# word_embeddings = data.word_embeddings()


def related_word_semantics_phrase(n1, n2):
    """True if the two phrases share a word with strongly related semantical similarity
       We filter words by postag and length, in order to avoid associations of too common words

    Returns:
        dict --
    """
    w1 = n1['value']
    w2 = n2['value']

    word_freq_w1 = wordfreq.word_frequency(w1, 'en')
    word_freq_w2 = wordfreq.word_frequency(w2, 'en')
    if word_freq_w1 == 0 or word_freq_w2 == 0:
        # The word is not present in the wordfreq vocabulary, we skip
        return {'outcome': False}

    global word_embeddings
    if w1 not in word_embeddings or w2 not in word_embeddings:
        # We do not have an embedding for this word, we skip
        return {'outcome': False}

    semantical_similarity = word_embeddings.similarity(w1, w2)

    semantical_similarity = float(semantical_similarity)

    if semantical_similarity < 0.4:
        # If the semantical similarity is lower that this threshold, we skip
        return {'outcome': False}

    word_score_w1 = -math.log(word_freq_w1)
    word_score_w2 = -math.log(word_freq_w2)

    word_freq_score = (word_score_w1+word_score_w2)/2

    # We compute the final score of the association as the Semantical similarity weighted by the unfrequency score
    similarity = word_freq_score*semantical_similarity

    thr_similarity = 4.0
    if similarity >= thr_similarity:
        return {'outcome': True,
                'word_1': w1,
                'word_2': w2}
    else:
        return {'outcome': False}

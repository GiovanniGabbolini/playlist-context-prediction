from src.text_processing.preprocess_phrase import tokenize
from src.text_processing.preprocess_word import lower
import wordfreq
import re

thr = 1e-6


def uncommon_words(n1, n2):
    uncommon_words_found = [[], []]
    for idx, n in enumerate([n1, n2]):
        words = tokenize(n['value'], funcs_word=[lower])

        # Filter out words based on their lengths and if they do not contain any letter
        filtered_words = []
        for w in words:
            if len(w) > 3 and re.search("[a-zA-Z]", w):
                filtered_words.append(w)

        words_freqs = {}
        for w in filtered_words:
            if w not in words_freqs:
                probability = wordfreq.word_frequency(
                    w, 'en', wordlist='large')
                words_freqs[w] = probability

        res = [key for key in words_freqs.keys()
               if words_freqs[key] < thr]
        uncommon_words_found[idx] = res

    shared_uncommon_words = set(
        uncommon_words_found[0]) & set(uncommon_words_found[1])
    if len(shared_uncommon_words) > 0:
        return {'outcome': True,
                'words': sorted(list(shared_uncommon_words), key=len, reverse=True),
                }
    else:
        return {'outcome': False}

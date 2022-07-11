import numpy as np


def get_word_frequencies(docs, labels, preprocessor):
    """Count the number of times each word in each doc is associated to
       positive (1) or negative (0) sentiment"""
    ys_list = np.squeeze(labels).tolist()
    freqs = {}

    for y, doc in zip(ys_list, docs):
        for token in preprocessor.preprocess(doc):
            pair = (token, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs


def lookup(freqs, word, label):
    for pair in freqs.keys():
        if (word, label) == pair:
            return freqs[(word, label)]
    return 0

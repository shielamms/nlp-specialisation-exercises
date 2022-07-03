import numpy as np

from modules.utils import lookup


class Classifier(object):
    def __init__(self):
        pass


class NaiveBayesClassifier(Classifier):
    def __init__(self):
        self.log_prior = None
        self.log_likelihood = {}

    def train(self, freqs, X, y):
        # N_pos = total frequencies of words in the positive class
        # N_get = total frequencies of words in the negative class
        N_pos = N_neg = 0
        for w_class_pair in freqs.keys():
            if w_class_pair[1] == 1:
                N_pos += freqs[w_class_pair]
            else:
                N_neg += freqs[w_class_pair]

        # D = total number of training docs
        # D_pos = total number of positive docs in training
        # D_neg = total number of negative docs in training
        D = len(y)
        D_pos = sum(y)
        D_neg = D - D_pos

        # log_prior = the log of the ratios between positive and negative docs
        # log(pos/neg) == log(pos) - log(neg)
        self.log_prior = np.log(D_pos) - np.log(D_neg)

        # V = number of unique words in the vocabulary
        vocab = [f[0] for f in freqs.keys()]
        V = len(set(vocab))

        for word in vocab:
            # frequency of the word appearing in positive docs
            w_freq_pos = lookup(freqs, word, 1)

            # frequency of the word appearing in negative docs
            w_freq_neg = lookup(freqs, word, 0)

            # probability of the word appearing in each class:
            # P(w,class) = freq(w,class) / N_class
            # Use laplacian (additive) smoothing to avoid 0 probability
            p_w_pos = (w_freq_pos + 1) / (N_pos + V)
            p_w_neg = (w_freq_neg + 1) / (N_neg + V)

            # log_likelihood = the log of the ratios of the probabilities
            # of positive and negative docs
            # log(pos/neg) == log(pos) - log(neg)
            self.log_likelihood[word] = np.log(p_w_pos) - np.log(p_w_neg)


    def predict(self, processed_doc):
        # always add the log_prior
        probability = 0
        probability += self.log_prior

        for word in processed_doc:
            if word in self.log_likelihood:
                probability += self.log_likelihood[word]

        if probability > 0:
            return 1
        else:
            return 0

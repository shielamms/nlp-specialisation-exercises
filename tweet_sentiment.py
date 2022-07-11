import random

import nltk
import numpy as np
from nltk.corpus import twitter_samples
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from modules.classifier import NaiveBayesClassifier
from modules.preprocessor import TweetPreprocessor
from modules.utils import get_word_frequencies

nltk.download('twitter_samples')


def tweet_sentiment():
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    print('Positive tweets: ', len(positive_tweets))
    print('Negative tweets: ', len(negative_tweets))

    # use 80% of the tweets for training
    train_pos = positive_tweets[:int(len(positive_tweets)*0.8)]
    train_neg = negative_tweets[:int(len(negative_tweets)*0.8)]
    val_pos = positive_tweets[int(len(positive_tweets)*0.8):]
    val_neg = negative_tweets[int(len(negative_tweets)*0.8):]

    train_x = train_pos + train_neg
    train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))

    val_x = val_pos + val_neg
    val_y = np.append(np.ones(len(val_pos)), np.zeros(len(val_neg)))

    # Preprocessing
    print('preprocessing...')
    tokenizer = TweetTokenizer(preserve_case=False,
                               strip_handles=True,
                               reduce_len=True)
    stemmer = PorterStemmer()
    preprocessor = TweetPreprocessor(tokenizer=tokenizer,
                                     stemmer=stemmer)

    freqs = get_word_frequencies(train_x, train_y, preprocessor)

    # Training
    print('training...')
    classifier = NaiveBayesClassifier()
    classifier.train(freqs, train_x, train_y)

    # Validation
    val_accuracy = validate(val_x, val_y, preprocessor, classifier)
    print('Validation accuracy: %0.4f' % val_accuracy)


def validate(val_x, val_ys, preprocessor, classifier):
    """
    Parameters:
        val_x: The list of tweets for validation
        val_ys: The list of correct classes of tweets in val_x
        preprocessor: a TweetPreprocessor object for cleaning the validation set
        classifier: a Classifier object to predict classes of the validation set
    Returns:
        accuracy: A float of the mean absolute error between predicted
                  and expected values
    """
    y_hats = []
    accuracy = 0

    for tweet in val_x:
        tweet_tokens = preprocessor.preprocess(tweet)
        y_hats.append(classifier.predict(tweet_tokens))

    # Use mean absolute error as the cost function
    error = np.mean(np.abs(y_hats - val_ys))
    accuracy = 1 - error

    return accuracy


def test_samples(preprocessor, classifier):
    samples = ['I am happy',
               'I am sad.',
               'This is not correct!',
               'This is awesome :D',]

    predictions = []
    for tweet in samples:
        tweet_tokens = preprocessor.preprocess(tweet)
        print(tweet, tweet_tokens)
        predictions.append(classifier.predict(tweet_tokens))

    print(predictions)


def lookup(freqs, word, label):
    """
    Parameters:
        freqs: A dictionary of frequencies of (word, class) pairs
        word: The word to look up in the freqs dictionary
        label: The class to look up in the freqs dictionary
    Returns:
        The value of the (word,class) pair in the freqs dictionary;
        If the pair is not found in the dictionary, then 0.
    """
    for pair in freqs.keys():
        if (word, label) == pair:
            return freqs[(word, label)]
    return 0


if __name__ == '__main__':
    tweet_sentiment()
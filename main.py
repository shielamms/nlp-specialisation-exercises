import random

import nltk
import numpy as np
from nltk.corpus import twitter_samples
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from modules.preprocessor import TweetPreprocessor

nltk.download('twitter_samples')


def main():
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    print('Positive tweets: ', len(positive_tweets))
    print('Negative tweets: ', len(negative_tweets))

    labels = np.append(np.ones(len(positive_tweets)),
                       np.zeros(len(negative_tweets)))

    freqs = get_word_frequencies(positive_tweets + negative_tweets, labels)
    print(freqs)

    # sample
    # tweet = positive_tweets[random.randint(0, len(positive_tweets))]
    # tweet_tokens = preprocess_tweet(tweet)
    # print(tweet_tokens)


def preprocess_tweet(tweet):
    tokenizer = TweetTokenizer(preserve_case=False,
                               strip_handles=True,
                               reduce_len=True)
    stemmer = PorterStemmer()
    preprocessor = TweetPreprocessor(tokenizer=tokenizer,
                                     stemmer=stemmer)
    return preprocessor.preprocess(tweet)


def get_word_frequencies(tweets, labels):
    """Count the number of times each word in each tweet is associated to
       positive (1) or negative (0) sentiment"""
    ys_list = np.squeeze(labels).tolist()
    freqs = {}

    for y, tweet in zip(ys_list, tweets):
        for token in preprocess_tweet(tweet):
            pair = (token, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs


if __name__ == '__main__':
    main()  # temp
import random

import nltk
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

    # sample
    tweet = positive_tweets[random.randint(0, len(positive_tweets))]
    tweet_tokens = preprocess_tweet(tweet)
    print(tweet_tokens)


def preprocess_tweet(tweet):
    tokenizer = TweetTokenizer(preserve_case=False,
                               strip_handles=True,
                               reduce_len=True)
    stemmer = PorterStemmer()
    preprocessor = TweetPreprocessor(tokenizer=tokenizer,
                                     stemmer=stemmer)
    return preprocessor.preprocess(tweet)


if __name__ == '__main__':
    main()  # temp
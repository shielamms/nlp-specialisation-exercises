import re
import string

import nltk
from nltk.corpus import stopwords as sw
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


class TweetPreprocessor():
    def __init__(self,
                 tokenizer=TweetTokenizer(),
                 stemmer=PorterStemmer()):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.stopwords = sw.words('english')

    def preprocess(self, tweet):
        # remove stock market tickers like $GE
        tweet = re.sub(r'\$\w*', '', tweet)
        # Remove "RT" from old retweets
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        # Remove hyperlinks
        tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
        # Remove hashtags
        tweet = re.sub(r'#', '', tweet)
        tokens = self.tokenizer.tokenize(tweet)
        clean_tokens = []

        for token in tokens:
            if (token not in self.stopwords and
                token not in string.punctuation):
                    stem = self.stemmer.stem(token)
                    clean_tokens.append(stem)

        return clean_tokens

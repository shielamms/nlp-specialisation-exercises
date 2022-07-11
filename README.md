# nlp-specialisation-exercises

This repository contains jupyter notebooks and python modules I wrote following the exercises in the [Natural Language Processing Course by DeepLearning.AI in Coursera](https://coursera.org/share/28c70a033bdce6bad5e65e3ddbb563be).

_This readme is very much a work in progress_

## What's inside?

Instead of uploading my Jupyter notebooks for the course exercises, I decided to modularise my code to make it reusable and to really let the concepts sink into my head while I write the code in different ways.

1. `tweet_sentiment.py`:
This project was derived from Week 2 of the course: Sentiment Analysis with Naïve Bayes.
The programming assignment is about preprocessing tweets from the nltk `twitter_samples` dataset, and training a Naive Bayes classifier to predict the sentiment of a tweet (1 if positive, 0 if negative). I'll keep on adding to this (along with more complete documentation) in the next few weeks.

2. `word_prediction.py`:
This project was derived from Week 3 of the course: Vector Space Models.
The programming assignment is about predicting the country name given a city. More generally, this predicts the 4th word given 3 words, where there is an association between the first 2 words. For example, if word1="Turkey", word2="Ankara", and word3="Bangkok", then this will predict the country of Bangkok using the association between "Ankara" and "Turkey" based on a distance metric of their respective embeddings. This project also contains my own definition of a PCA class, which reduces the dimensions of a given set of features (in this case, a subset of the embeddings) by getting only a subset of its eigenvectors.


## How to run the code

All code in this project were written and tested in Python 3.9. To run a file in the root directory of this project, I recommend creating a virtual environment, like this in pyenv and virtualenv:

```
pyenv local 3.9.2
python -m virtualenv venv
source venv/bin/activate
```

Then install the dependencies:

```
python -m pip install -r requirements.txt
```

From there, it's eazy peazy to run a project:

```
python tweet_sentiment.py
```

or

```
python word_prediction.py
```


### Expected Output for `tweet_sentiment.py`

Right now it's just a bunch of print statements to tell you how many positive and negative tweets there are in total, as well as the accuracy of the classifier on the validation set.
I trained the model on 80% of the dataset, and validated on the remaining 10%.

```
Positive tweets:  5000
Negative tweets:  5000
preprocessing...
training...
Validation accuracy: 0.9955
```

### How to use your own sample tweets to test the classifier

In `main.py`, there's a function called `test_samples()`. You can add your own sample tweets in the `samples` variable, then call `test_samples()` on `main()`. When you run `main.py`, this will use the classifier on each sample tweet, then output the predictions as an ordered list of 1s (positive) and 0s (negative).

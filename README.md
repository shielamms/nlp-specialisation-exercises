# nlp-specialisation-exercises

This repository contains jupyter notebooks and python modules I wrote following the exercises in the [Natural Language Processing Course by DeepLearning.AI in Coursera](https://coursera.org/share/28c70a033bdce6bad5e65e3ddbb563be).

_This readme is very much a work in progress_

## What's inside?

I'm currently in Week 2 of Course 1: NLP with Classification in Vector Spaces.
The programming assignment is about preprocessing tweets from the nltk `twitter_samples` dataset, and training a Naive Bayes classifier to predict the sentiment of a tweet (1 if positive, 0 if negative). I'll keep on adding to this (along with more complete documentation) in the next few weeks.

Instead of uploading my Jupyter notebooks for the exercises, I decided to modularise the code to make it reusable and to really let the concepts sink into my head while I write the code in different ways.

## How to run the code

This code was written and tested in Python 3.9. To run this code, I recommend creating a virtual environment, like this in pyenv and virtualenv:

```
pyenv local 3.9.2
python -m virtualenv venv
source venv/bin/activate
```

Then install the dependencies:

```
python -m pip install -r requirements.txt
```

From there, it's eazy peazy to run the code:

```
python main.py
```

### Expected Output

Right now it's just a bunch of print statements to tell you how many positive and negative tweets there are in total, as well as the accuracy of the classifier on the validation set.
I trained the model on 80% of the dataset, and validated on the remaining 10%.

```
Positive tweets:  5000
Negative tweets:  5000
preprocessing...
training...
Validation accuracy: 0.9955
```

## How to use your own sample tweets to test the classifier

In `main.py`, there's a function called `test_samples()`. You can add your own sample tweets in the `samples` variable, then call `test_samples()` on `main()`. When you run `main.py`, this will use the classifier on each sample tweet, then output the predictions as an ordered list of 1s (positive) and 0s (negative).

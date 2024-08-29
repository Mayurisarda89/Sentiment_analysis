# Sentiment_analysis
This is a sentiment analysis application developed using Streamlit and the Naive Bayes Classifier from the NLTK library. The application analyzes the sentiment of tweets, determining whether they are positive or negative.

## Overview
This application utilizes the NLTK (Natural Language Toolkit) to perform sentiment analysis on tweets. It leverages a Naive Bayes Classifier trained on a dataset of positive and negative tweets to classify user-inputted text as either "Positive" or "Negative."

## Dataset
The dataset used in this application comes from the twitter_samples corpus provided by NLTK. It includes:

5,000 positive tweets from positive_tweets.json.
5,000 negative tweets from negative_tweets.json.
These tweets are tokenized, cleaned to remove noise (e.g., URLs, mentions, and punctuation), and then lemmatized (i.e., reduced to their base forms). 

## Model Training and accuracy 
The cleaned and tokenized tweets are used to train a Naive Bayes Classifier. The dataset is split into:

7,000 tweets for training the model.
3,000 tweets for testing the model.
The classifier achieves an accuracy of 99.5% on the test data, indicating a high level of reliability in predicting the sentiment of tweets.

## Code Summary
Data Preprocessing: The tweets are tokenized, cleaned, and lemmatized to prepare them for the model.
Model Training: The processed tweets are used to train a Naive Bayes Classifier.
Sentiment Analysis: The trained model is used to predict the sentiment of new tweets entered by the user.
User Interface: A simple UI is provided through Streamlit, allowing users to input a tweet and see the predicted sentiment.




# general purpose imports
import re
import io

# ORM related imports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_, extract
from create_schema import User, Tweet, Tag

# nltk related imports
import nltk
from nltk.corpus import webtext
from nltk.corpus import stopwords
from nltk.probability import FreqDist
# - this is our sentence tokenizer (pre-trainerd instance of PunktSentenceTokenizer)
from nltk.tokenize import sent_tokenize

# connect to db
engine = create_engine('sqlite:///tweets.db')
session = sessionmaker(bind=engine)()

# global stop words var
english_stops = stopwords.words('english')

# variable to store training set of tweets
training_tweets = []

def load_training_set():
  # opening file with training data
  with io.open('data.txt', 'r', encoding='utf16') as f:
    # reading line by line
    for line in f:
      # split string into list using space as separator
      string_in_array = filter(None, line.strip().split(' '))

      # first element of every list(converted string) is either 0 or 1 (negative or positive)
      # our training data needs to look like:
      # [(['token1','token2','token3'], 'positive'), (...), ... , (...)]
      lowercase_array_of_tweet_words = []
        
      for w in string_in_array[1:]:
        if len(w) >= 3:
          lowercase_array_of_tweet_words.append(w.lower())
                
      sent_word = 'positive'

      if string_in_array[0] == '0':
        sent_word = 'negative'
      
      tuple = (lowercase_array_of_tweet_words, sent_word)
      training_tweets.append(tuple)

# return all words from all tweets in a single list 
def get_words_in_tweets(tweets):
  all_words = []
  for (words, sentiment) in tweets:
    all_words.extend(words)
  return all_words

# builds freq distribution in dictionary and return list of keys of the distribution
def get_word_features(wordlist):
  wordlist = nltk.FreqDist(wordlist)
  word_features = wordlist.keys()
  return word_features

def extract_features(document):
  document_words = set(document)
  features = {}
  for word in get_word_features(get_words_in_tweets(training_tweets)):
      features['contains(%s)' % word] = (word in document_words)
  return features

def build_classifier():
  training_set = nltk.classify.apply_features(extract_features, training_tweets)
  classifier = nltk.NaiveBayesClassifier.train(training_set)

  return classifier

# this function prepares tweet to be classified, i.e
# makes sure spaces between tags and words are removed etc
def clean_up_tweet_text(text):
  # this sub removes space between @ or # and tag name
  text = re.sub(r"([@#])\s", r"\1", text)
  # this sub fixes URLs (e.g. removes space in 'http(s):// www...' and 'http(s)://www.') 
  text = re.sub(r"(http(s)?:\/\/(www\.)?)\s", r"\1", text)
  # more subs can be added in the similar fashion
  return text

def process_tweet(tweet):
  print
def classify_tweet(tweet):
  print

def main():
  load_training_set()
  classifier = build_classifier()

  # 4 months in a cycle
  for month in [4,5,6,7]:
    # complex select of tweets
    # in our case - tweets that contain #genomic and do not contain the other five tags below
    tweets = session.query(Tweet).join(Tweet.tags) \
      .filter(
        and_(
          Tag.name == '#genomic',
          ~Tag.name.in_(['#23andme', '#precisionmedicine', '#personalizedmedicine', '@GenomicsEngland', '#datasharing'])
        )  
      ) \
      .filter(
        and_(
          extract('month', Tweet.timestamp) == month), 
          extract('year', Tweet.timestamp) == 2016
        )

    # this is where we store classified tweets
    positive = []
    negative = []

    # iterating tweets
    for t in tweets:
      # clean up every tweet, modify the clean_up_tweet_text to add more features      
      tweet_text = clean_up_tweet_text(t.tweet_text)
      
      # classification
      sentiment = classifier.classify(extract_features(tweet_text.split()))

      if sentiment == 'positive':
        # save to positive list as tuple with id and text
        positive.append((t.id, t.tweet_text))
      else:
        # -//- negative -//-
        negative.append((t.id, t.tweet_text))
    
    # output
    print "month: " + str(month)
    print "positive tweets: " + str(len(positive))
    print "positive tweets: " + str(len(negative))
    print ""

if __name__ == '__main__':
   main()
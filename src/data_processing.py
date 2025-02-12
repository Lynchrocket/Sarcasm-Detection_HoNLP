import os
import re
import emoji.unicode_codes
import numpy as np
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
import contractions
import emoji

from . import utils

cur_dir = os.getcwd()

def get_dictionary():
    dictionary_file = f'{cur_dir}/data/word_list.txt'
    dictionary = utils.load_file(dictionary_file)
    return dictionary

def get_stopwords():
    stopwords_file = f'{cur_dir}/data/stopwords.txt'
    stopwords = utils.load_file(stopwords_file)
    return stopwords

def camel_split(term):
    # add white space before every number
    term = re.sub(r'([0-9]+)', r' \1', term)
    # add white space before every ordinal number
    term = re.sub(r'([0-9]+(?:st|nd|rd|th)?)', r' \1', term)
    splits = re.findall(r'[a-z]+|[A-Z][a-z]*|[A-Z]+(?=\b)|[0-9]+(?:st|nd|rd|th)?', term)
    return splits

def process_hashtag(hashtag, dictionary):
    """
    split a hashtag into words
    """
    hashtag = hashtag[1:]
    if hashtag != hashtag.lower() and hashtag != hashtag.upper():
        return camel_split(hashtag)
    else:
        split = []
        i = 0
        while i <= len(hashtag):
            loc = i
            for j in range(i+1, len(hashtag)+1):
                if hashtag[i:j].lower() in dictionary:
                    loc = j
            if i == loc:
                i += 1
            else:
                split.append(hashtag[i:loc])
                i = loc
        return split
        
def raw_clean(tweet):
    dictionary = get_dictionary()
    # Add white space before every punctuation sign so that we can split around it and keep it
    tweet = re.sub('([!?*&%"~`^+{}])', r' \1 ', tweet)
    tweet = re.sub('\s{2,}', ' ', tweet)
    words = tweet.split()
    valid_words = []
    for word in words:
        # remove #sarca* hashtags
        if word.startswith('#sarca'):
            continue
        # replace user mentions with @user
        if word.startswith('@'):   
            word = '@user'
        # remove urls
        if 'http' in word:
            continue
        # process hashtags
        if word.startswith('#'):
            word = process_hashtag(word, dictionary)
        if isinstance(word, str):
            valid_words.append(word)
        else:
            valid_words.extend(word)
    return ' '.join(valid_words)


def grammatical_clean(tweet, stopword=False):
    """
    clean a tweet by expanding contractions, and lemmatizing
    """
    stopwords = get_stopwords()
    lemmatizer = WordNetLemmatizer()
    words = tweet.split()
    valid_words = []
    for word in words:
        if stopword and word.lower() in stopwords:
            continue
        
        # process emoji
        if emoji.is_emoji(word):
            emoji_translation = emoji.demojize(word)[1:-1].split('_')
            valid_words.extend(emoji_translation)
            continue
        # reveal abbreviations and contractions
        # abbreviations: asap -> as soon as possible; idk -> i don't know
        # contractions: I'm -> I am; don't -> do not
        expanded_words = contractions.fix(word).split()
        for expanded in expanded_words:
            # Cannot do lemmatization without changing the case
            # https://stackoverflow.com/questions/8003003/nltk-lemmatizer-doesnt-know-what-to-do-with-the-word-americans
            lemmatized = lemmatizer.lemmatize(expanded.lower(), 'v') # for verb
            lemmatized = lemmatizer.lemmatize(lemmatized, 'n') # for noun
            if expanded.isupper():
                lemmatized = lemmatized.upper()
            elif expanded.istitle():
                lemmatized = lemmatized.title()
            valid_words.append(lemmatized)

    return ' '.join(valid_words)

def clean_tweet(tweet):
    tweet = raw_clean(tweet)
    tweet = grammatical_clean(tweet)
    return tweet

def get_clean_data(train_filename, test_filename, save=False): 
    train_data = utils.load_csv_data(train_filename, ['Set', 'Label', 'Text'], sep='\t+')
    train_tweets, train_labels = train_data['Text'], train_data['Label'].apply(int)

    test_data = utils.load_csv_data(test_filename, ['Set', 'Label', 'Text'], sep='\t+')
    test_tweets, test_labels = test_data['Text'], test_data['Label'].apply(int)

    train_tweets = train_tweets.apply(clean_tweet)
    test_tweets = test_tweets.apply(clean_tweet)

    if save:
        utils.save_file(f'{cur_dir}/processed_data/train_tweets_clean.txt', train_tweets)
        utils.save_file(f'{cur_dir}/processed_data/test_tweets_clean.txt', test_tweets)

    return train_tweets, train_labels, test_tweets, test_labels


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))    

    train_filename = f'{cur_dir}/data/datasets/ghosh/train_sample.txt'
    test_filename = f'{cur_dir}/data/datasets/ghosh/test_sample.txt'

    train_tweets, train_labels, test_tweets, test_labels = get_clean_data(train_filename, test_filename, save=True)

    print(train_tweets.head())
    print(test_tweets.head())
    print(train_labels.head())
    print(test_labels.head())


from collections import Counter, OrderedDict
from operator import itemgetter

import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

import re
from bs4 import BeautifulSoup


def review_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()

    text = BeautifulSoup(review, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())  # Convert to lower case
    words = text.split()  # Split string into words
    words = [w for w in words if w not in stopwords.words(
        "english")]  # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words]  # stem

    return words


def build_dict(data, vocab_size=5000):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""

    # TODO: Determine how often each word appears in `data`. Note that `data` is a list of sentences and that a
    #       sentence is a list of words.
    word_counter = Counter()
    for sentence in data:  # for each sentence
        # count the frequency of each word in each sentence
        word_counter.update(sentence)

    # A dict storing the words that appear in the reviews along with how often they occur
    word_count = dict(word_counter.most_common())

    # TODO: Sort the words found in `data` so that sorted_words[0] is the most frequently appearing word and
    #       sorted_words[-1] is the least frequently appearing word.
    # word_count is already in the order (the most frequently to least frequently)
    sorted_words = list(word_count.keys())
    print("sorted_words\n" + str(sorted_words) + "\n")

    word_dict = {}  # This is what we are building, a dictionary that translates words into integers
    # The -2 is so that we save room for the 'no word'
    for idx, word in enumerate(sorted_words[:vocab_size - 2]):
        # 'infrequent' labels
        word_dict[word] = idx + 2

    return word_dict

def build_dict2(data, vocab_size=5000):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""

    # TODO: Determine how often each word appears in `data`. Note that `data` is a list of sentences and that a
    #       sentence is a list of words.
    word_count = {}
    for sentence in data:  # for each sentence
        # wordlist = sentence.split()
        # count the frequency of each word in each sentence
        for word in sentence:
            word_count[word] = word_count.setdefault(word, 0) + 1
            # if word not in word_count:
            #     print(word)
            #     word_count[word] += 1

    # A dict storing the words that appear in the reviews along with how often they occur
    

    # TODO: Sort the words found in `data` so that sorted_words[0] is the most frequently appearing word and
    #       sorted_words[-1] is the least frequently appearing word.
    # word_count is already in the order (the most frequently to least frequently)
    sorted_words = []
    sorted_tuple = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    print("sorted_tuple\n" + str(sorted_tuple) + "\n")
    for sorted_word in sorted_tuple:
        # print("sorted_word\n" + str(sorted_word[0]) + "\n")
        sorted_words.append(sorted_word[0])
    

    word_dict = {}  # This is what we are building, a dictionary that translates words into integers
    # The -2 is so that we save room for the 'no word'
    for idx, word in enumerate(sorted_words[:vocab_size - 2]):
        # 'infrequent' labels
        word_dict[word] = idx + 2

    return word_dict



train_X = [['steve', 'carrel', 'prove', 'great', 'lead', 'man', 'wonder', 'origin', 'raunchi', 'breath', 'fresh', 'air', 'wet', 'geniusli', 'hilari', 'basic', 'movi', 'titl', 'say', 'andi', 'stitzer', '40', 'year', 'old', 'male', 'work', 'electron', 'store', 'bit', 'nerd', 'love', 'videogam', 'comic', 'biggest', 'collect', 'peer', 'work', 'store', 'find', 'virgin', 'rather', 'sex', 'dialogu', 'fill', 'poker', 'game', 'andi', 'go', 'rather', 'funni', 'hell', 'odyessi', 'rude', 'sexual', 'awaken', 'alway', 'screw', 'lead', 'lose', 'virgin', 'eventu', 'get', 'lucki', 'end', 'leav', 'littl', 'one', 'home', 'take', 'entir', 'famili', 'see', 'awesom', 'romant', 'adult', 'comedi', 'hook', 'crack', 'begin', 'time', 'wish', 'wore', 'extra', 'thick', 'absorb',
            'undergar', 'thing', 'say', 'bad', 'steve', 'carrel', 'recogn', 'leav', 'man', '20', 'year', 'ago', 'definit', 'gonna', 'win', 'best', 'breakthrough', 'male', 'perform', 'next', 'year', 'mtv', 'movi', 'award', 'bet', 'hard', 'earn', 'dollar', 'peopl', 'give', 'one', 'perfect', '10'], ['awesom', 'improb', 'foolish', 'potboil', 'least', 'redeem', 'crisp', 'locat', 'photographi', 'unbeliev', 'gener', 'much', 'way', 'tension', 'kinda', 'hope', 'stanwyck', 'make', 'back', 'time', 'realli', 'saddl', 'wet', 'way', 'one', 'husband', 'idiot', 'child', 'well', 'run', 'meeker', 'nag', 'question', 'remain', 'sort', 'wood', 'pier', 'support', 'made', 'rotten', 'piec', 'pull', 'float', 'stanwyck', 'alway', 'impecc', 'profession', 'best', 'could', 'materi', 'threadbar']]

word_dict = build_dict(train_X)

print("word_dict\n" + str(word_dict) + "\n")

word_dict2 = build_dict2(train_X)

print("word_dict2\n" + str(word_dict2) + "\n")

first2pairs = list(word_dict2.keys())[:2]
# word_dict2
# first5vals = [v for v in word_dict2.keys()[:2]]
print("first2pairs\n" + str(first2pairs) + "\n")
import re
import pandas as pd
import string
string.punctuation

import sys
sys.path.append("../")

from common_utils import space_special_chars, segment_hashtags, emoji


def read_lexicon(lexicon_path):
    with open(lexicon_path, 'r', encoding="utf-8") as f:
        lines = f.read().splitlines()
    return lines

def preprocessing_offensive_wrds(offensive_list):
    for word in offensive_list:
        word = space_special_chars(word)
        word = str(word).lower()
        word = word.translate(str.maketrans('', '', string.punctuation))
        word = re.sub(r'\u300c', '', word)
    return offensive_list


def filter_offensive_words(snt, offensive_list):
    wrds = snt.split()
    off_wrds = []
    for wrd in wrds:
        if not wrd.isalnum():
            continue
        if wrd.lower() in offensive_list:
            off_wrds.append(wrd)

    return off_wrds


def read_corpus(filename = "data/train.tsv", delimiter = "\t", offensive_lexiconpath = '../../data_creation/final_offensive_lexicon.txt'):
    df = pd.read_csv(filename, sep=delimiter, names = ["text", "label"])

    df['preprocessed_text'] = df['text'].str.replace(r'&amp;', r'and', regex=True)

    # Hashtags
    df['preprocessed_text'] = df['preprocessed_text'].apply(segment_hashtags)
    df['preprocessed_text'] = df['preprocessed_text'].str.lower()

    df['preprocessed_text'] = df['preprocessed_text'].apply(space_special_chars)
    df['preprocessed_text'] = df['preprocessed_text'].str.replace(r' +', r' ', regex=True)

    # remove urls
    df['preprocessed_text'] = df['preprocessed_text'].str.replace(r'URL', "http", regex=True)

    # removal of @USER token
    df['preprocessed_text'] = df['preprocessed_text'].str.replace(r'(@USER\s*){4,}','@USER @USER @USER ', regex=True)

    # Emoji to natural language
    df['preprocessed_text'] = df['preprocessed_text'].apply(emoji.demojize)
    df['preprocessed_text'] = df['preprocessed_text'].str.replace(r':(\w+):', r'\g<1>', regex=True)
    df['preprocessed_text'] = df['preprocessed_text'].str.replace(r'_', r' ', regex=True)
    df['preprocessed_text'] = df['preprocessed_text'].str.replace(r':', r' ', regex=True)

    df['preprocessed_text'] = df['preprocessed_text'].str.replace(r' +', r' ', regex=True)

    offensive_list = read_lexicon(offensive_lexiconpath)
    offensive_list = preprocessing_offensive_wrds(offensive_list)
    df['filtered_words'] =  df['preprocessed_text'].apply(lambda x: filter_offensive_words(x, offensive_list))

    df[["preprocessed_text", "label", "filtered_words"]].to_csv(filename+'.check')
    reviews = df['preprocessed_text'].values.tolist()
    labels = df['label'].values.tolist()
    filtered_words = df['filtered_words'].values.tolist()

    labels_w_filtered_words = list(zip(labels, filtered_words))

    return reviews, labels_w_filtered_words


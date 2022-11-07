import pandas as pd
import sys
sys.path.append("../")
from common_utils import space_special_chars, segment_hashtags, emoji


def read_corpus(filename = "data/train.tsv", delimiter = "\t"):
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

    df[["preprocessed_text", "label"]].to_csv(filename+'.check')
    reviews = df['preprocessed_text'].values.tolist()
    labels = df['label'].values.tolist()

    return reviews, labels

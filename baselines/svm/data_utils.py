import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import emoji
from nltk.corpus import stopwords
import nltk
import spacy
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def spacy_pos(txt, nlp):
  #pos tagging
  return [token.pos_ for token in nlp(txt)]


def download_nltk_resources():
  nltk.download('omw-1.4')
  nltk.download('punkt')
  nltk.download('wordnet')
  nltk.download('stopwords')

download_nltk_resources()

"""# SET UP - DATA"""
def read_corpus(PATH):
  #read data
  df = pd.read_csv(PATH,sep='\t', names=["text", "label"])
  return df

def stop_words_rem(data):
  #remove stop words
  stop_words = set(stopwords.words('english'))
  data_clean = []
  for word in data:
    if word not in stop_words:
      data_clean.append(word)
  return data_clean

def lemmatize(data):
  #lemmatization
  lemmatizer = WordNetLemmatizer()
  data_lem = []
  for word in data:
    lem_word = lemmatizer.lemmatize(word)
    data_lem.append(lem_word)
  return data_lem

def emoji_nl(data):
    #convert emojis to natural language
    data = emoji.demojize(data)
    data = re.sub(r':(\w+):', r'\g<1>', data)
    data = re.sub(r'_', r' ', data)
    return data

def emoji_remove(data):
  #remove emojis with no conversion
  emojis = re.compile("["
          u"\U0001F600-\U0001F64F"
          u"\U0001F300-\U0001F5FF"
          u"\U0001F680-\U0001F6FF"
          u"\U0001F1E0-\U0001F1FF"
                           "]+", flags=re.UNICODE)
  emojis_removed = emojis.sub(r'', data)
  return emojis_removed

def preprocessing(inp):
    '''main preprocessing function'''
    #inp = emoji_remove(str(inp)) #emoji removal
    inp = emoji_nl(inp) #emoji to nl
    inp = str(inp).lower()  # lowercase
    inp = re.sub(r"\s{2,}", " ", inp)
    inp = re.sub(r'@user', '', inp) #remove user
    #inp = re.sub(r'url', ' ', inp) #keeping url gives better results
    #data = re.sub(r'#', '', inp) #remove hashtag symbol
    inp = re.sub(r'#([a-zA-Z0-9_]{1,50})', '' , inp) #remove entire hashtag
    inp = re.sub(r'\d+', '[NUM]', inp) #num token instead of numbers - gives better results
    inp = inp.strip() #whitespace
    inp = word_tokenize(inp) #tokenization
    inp = stop_words_rem(inp) #gave better results
    inp = lemmatize(inp)
    return inp

"""# VECTORIZERS"""

def get_vectorizer(vectorizer_name):
  # Convert the texts to vectors
  # We use a dummy function as tokenizer and preprocessor,
  # since the texts are already preprocessed and tokenized.
  if vectorizer_name == "tfidf":
      #TF-IDF vectorizer
      vec = TfidfVectorizer(preprocessor=preprocessing, tokenizer=preprocessing, min_df=2)
  elif vectorizer_name == "countvec":
      # Bag of Words vectorizer
      vec = CountVectorizer(preprocessor=preprocessing, tokenizer=preprocessing, min_df=2)
  elif vectorizer_name == "bothvec":
      # TF-IDF and Bag of Words vectorizer
      tf_idf = TfidfVectorizer(preprocessor=preprocessing, tokenizer=preprocessing, min_df=2)
      count = CountVectorizer(preprocessor=preprocessing, tokenizer=preprocessing, min_df=2)
      vec = FeatureUnion([("count", count), ("tfidf", tf_idf)])
  elif vectorizer_name == "ngram_count":
      # Bag of Words vectorizer with N-grams
      vec = CountVectorizer(ngram_range=(1,3), min_df=3)
  elif vectorizer_name == "ngram_tfidf":
      # Bag of Words vectorizer with TF-IDF
      vec = TfidfVectorizer(ngram_range=(1,3), min_df=3)
  elif vectorizer_name == "ngram_bothvec":
      # Bag of Words vectorizer with TF-IDF and N-grams
      ngram = CountVectorizer(ngram_range=(1,3), min_df=3)
      tf_idf = TfidfVectorizer(ngram_range=(1,3), min_df=3)
      vec = FeatureUnion([("ngram", ngram), ("tfidf", tf_idf)])
  elif vectorizer_name == "pos":
      #Part of Speech Tagging
      nlp = spacy.load("en_core_web_sm")
      args = {"nlp": nlp}
      countvec = CountVectorizer()
      pos = CountVectorizer(tokenizer= lambda text: spacy_pos(text, **args))
      vec = FeatureUnion([("count", countvec), ("pos", pos)])
  return vec


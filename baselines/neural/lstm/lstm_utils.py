import numpy as np
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
import pickle
import sys
sys.path.append("../")
from utils import read_corpus

def create_vectorizer(max_seq_len, X_train, X_dev):
    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=max_seq_len)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    return vectorizer, voc


def vectorize_inputtext(vectorizer, list_of_texts):
    # Transform input to vectorized input
    inp_vect = vectorizer(np.array([[s] for s in list_of_texts])).numpy()
    return inp_vect

def read_testdata_andvectorize(test_filename, vectorizer, encoder):
    # Read in test set and vectorize
    X_test, Y_test = read_corpus(test_filename)
    tokens_test = vectorize_inputtext(vectorizer, X_test)
    Y_test_bin = encoder.fit_transform(Y_test)
    return X_test, Y_test, tokens_test, Y_test_bin

def load_vectorizer_pickle(filename):
    with open(filename, "rb") as fh:
        from_disk = pickle.load(fh)
    new_v = TextVectorization.from_config(from_disk['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    new_v.set_weights(from_disk['weights'])
    return new_v
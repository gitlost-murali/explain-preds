#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python

''' Trains a LSTM model using train & dev set. If test set is mentioned, accuracy/F1-score (Macro)
is calculated with an option of showing the confusion matrix.

To run this file,

1. One can simply run the file without any command line arguments because the script 
is set up with best settings as default values for arguments.

python lstm_baseline.py

Running with the best hyperparams must give a macro F1-score of around 89/90.

Or if you want to change any of the arguments, please type

python lstm_baseline.py --help
'''

import random as python_random
import json
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Embedding, LSTM, Dropout
from keras.initializers import Constant
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Bidirectional
# Make reproducible as much as possible

import sys
sys.path.append("../")

from utils import create_class_weights, read_data,\
                  test_set_predict, save_picklefile, numerize_labels
from lstm_utils import vectorize_inputtext, create_vectorizer

from gensim.models import KeyedVectors



def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='../../../data/train.tsv', type=str,
                        help="Input file to learn from (default train.txt)")
    
    parser.add_argument("-d", "--dev_file", type=str, default='../../../data/dev.tsv',
                        help="Separate dev set to read in (default dev.txt)")

    parser.add_argument("-t", "--test_file", type=str, default='../../../data/test.tsv',
                        help="If added, use trained model to predict on test set")

    parser.add_argument("-e", "--embeddings", default='embeddings/cc.en.300.vec', type=str,
                        help="Embedding file we are using (Strictly .vec files)")
    parser.add_argument("--show_confusionmatrix", default=True, type=bool,
                        help="Show confusion matrix of the model on test set")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", default=50, type=int,
                        help="Number of epochs for training")
    parser.add_argument("--max_seq_len", default=100, type=int,
                        help="Maximum length of input sequence after BPE")
    parser.add_argument("--seed", default=1234, type=int,
                        help="Seed for reproducibility")
    parser.add_argument("--output_modelname", default="models/best_model.h5", type=str,
                        help="Filename to save the best model")
    args = parser.parse_args()
    return args


# In[26]:

def read_embeddings(embeddings_file):
    '''Read in word embeddings from file and save as numpy array'''
    embeddings = json.load(open(embeddings_file, 'r'))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_emb_matrix(voc, emb):
    '''Get embedding matrix given vocab and the embeddings'''
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    embedding_vector = None
    for word, i in word_index.items():
        try:
            embedding_vector = emb[word]
        except:
            pass
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def create_model(args, Y_train, emb_matrix):
    '''Create the Keras model to use'''
    # Define settings, you might want to create cmd line args for them
    loss_function = 'sparse_categorical_crossentropy'
    optim = Adam(learning_rate=args.learning_rate)
    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = len(set(Y_train))

    # Now build the model
    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, 
                        embeddings_initializer=Constant(emb_matrix),trainable=True,
                        name="embedding_updatable"))
    # Here you should add LSTM layers (and potentially dropout)
    model.add(Dense(units=256, activation="relu", name="dense_256"))
    model.add(Dropout(0.2, name="dropout_0.2"))

    model.add(Bidirectional(LSTM(256, return_sequences=True),))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dropout(0.2, name="dropout2_0.2"))
    # Ultimately, end with dense layer with softmax
    model.add(Dense(units=num_labels, activation="softmax", name=f"dense_{num_labels}"))
    # Compile model using our settings, check for accuracy
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])

    print(model.summary())
    return model


def train_model(args, model, X_train, Y_train, X_dev, Y_dev, encoder, class_weights):
    '''Train the model here. Note the different settings you can experiment with!'''
    # Potentially change these to cmd line args again
    # And yes, don't be afraid to experiment!
    verbose = 1
    batch_size = args.batch_size
    epochs = args.num_epochs
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It's also possible to monitor the training loss with monitor="loss"
    earlystopping = EarlyStopping(monitor='val_loss', patience=3)
    mckpt = ModelCheckpoint(args.output_modelname, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    # Finally fit the model to our data
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, 
              callbacks=[earlystopping, mckpt], batch_size=batch_size, 
              validation_data=(X_dev, Y_dev), class_weight = class_weights)

    # Print final accuracy for the model (clearer overview)
    test_set_predict(model, X_dev, Y_dev, "bestmodel-on-dev", encoder)
    return model


def convert_traindev(vectorizer, X_train, X_dev):
    # Transform input to vectorized input
    X_train_vect = vectorize_inputtext(vectorizer, X_train)
    X_dev_vect = vectorize_inputtext(vectorizer, X_dev)
    return X_train_vect, X_dev_vect


def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()
    print(args)

    seednum = args.seed


    np.random.seed(seednum)
    tf.random.set_seed(seednum)
    python_random.seed(seednum)

    X_train, Y_train, X_dev, Y_dev = read_data(args)
    vectorizer, voc = create_vectorizer(args.max_seq_len, X_train, X_dev)

    kv = KeyedVectors.load_word2vec_format(args.embeddings, binary=False)
    emb_matrix = get_emb_matrix(voc, kv)
    # Create model
    model = create_model(args, Y_train, emb_matrix)
    encoder, Y_train_bin, Y_dev_bin = numerize_labels(Y_train, Y_dev)
    X_train_vect, X_dev_vect = convert_traindev(vectorizer, X_train, X_dev)

    class_weights = create_class_weights(Y_train, encoder)

    # Train the model
    model = train_model(args, model, X_train_vect, Y_train_bin,
                        X_dev_vect, Y_dev_bin, encoder, class_weights)

    save_picklefile(encoder, f"{args.output_modelname}.pickle")
    print(f"Label encoder is saved to {args.output_modelname}.pickle")


    #https://stackoverflow.com/questions/65103526/how-to-save-textvectorization-to-disk-in-tensorflow
    # Pickle the config and weights
    save_picklefile({'config': vectorizer.get_config(), 'weights': vectorizer.get_weights()}
                    , f"{args.output_modelname}.details.vec")
    print(f"Vectorizer saved to {args.output_modelname}.details")


if __name__ == '__main__':
    main()





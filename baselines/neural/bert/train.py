#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Trains a BERT model variants using train & dev set. If test set is mentioned, accuracy/F1-score (Macro)
is calculated with an option of showing the confusion matrix.
'''
import argparse
import random as python_random

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

import sys
sys.path.append("../")

from utils import create_class_weights, read_data,\
                  test_set_predict, save_picklefile, numerize_labels

from bert_utils import vectorize_inputtext, load_model

def create_arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--train_file", default='../../../data/train.tsv', type=str,
                        help="Input file to learn from (default train.txt)")
    
    parser.add_argument("-d", "--dev_file", type=str, default='../../../data/dev.tsv',
                        help="Separate dev set to read in (default dev.txt)")
    
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="Learning rate for the optimizer")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for training")

    parser.add_argument("--num_epochs", default=5, type=int,
                        help="Number of epochs for training")

    parser.add_argument("--max_seq_len", default=200, type=int,
                        help="Maximum length of input sequence after BPE")

    parser.add_argument("--langmodel_name", default="bert-base-uncased", type=str,
                        help="Name of the base pretrained language model")

    parser.add_argument("--output_modelname", default="models/bert-outputs", type=str,
                        help="Name of the trained model that will be saved after training")

    parser.add_argument("--seed", default=1234, type=int,
                        help="Seed for reproducible results")

    args = parser.parse_args()
    return args


def get_lr_metric(optimizer):
    '''
    Function for printing LR after each epoch
    https://stackoverflow.com/questions/47490834/how-can-i-print-the-learning-rate-at-each-epoch-with-adam-optimizer-in-keras
    '''
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr


def create_model(args, Y_train, learning_rate = 5e-5, lm = "bert-base-uncased",
                 batchsize = 16, num_epochs = 2):
    """
    Create model and tokenizer using the hf model reponame
    Args:
        args: Command line argument passed
        Y_train: Training label data for creating the final classifier layer. 
                 Only length used
        learning_rate (float): Learning rate. Defaults to 5e-5.
        lm (str, optional): Language model name. Defaults to "bert-base-uncased".
        batchsize (int, optional): Batch size for batching input. Defaults to 16.
        num_epochs (int, optional): Number of epochs. Defaults to 2.

    Returns:
        model and its tokenizer
    """
    model, tokenizer = load_model(lm, num_labels= len(set(Y_train)))
    loss_function = SparseCategoricalCrossentropy(from_logits=True)
    starter_learning_rate = learning_rate
    end_learning_rate = learning_rate/100
    decay_steps = (len(Y_train)/batchsize)*num_epochs
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                                    starter_learning_rate, decay_steps,
                                    end_learning_rate, power=0.5)
    optim = Adam(learning_rate=learning_rate_fn)
    lr_metric = get_lr_metric(optim)
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy', lr_metric])
    return model, tokenizer


def train_model(model, tokens_train, Y_train,
                tokens_dev, Y_dev, encoder, 
                class_weights, batchsize = 16,
                num_epochs = 2, best_modelname = "bert-outputs",
                ):
    """
    Train the model here
    And also validate for every epoch
    EarlyStopping is implemented here with patience of 3 epochs
    Returns:
        model: Trained model
    """
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It's also possible to monitor the training loss with monitor="loss"
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    chkpt_calllback = tf.keras.callbacks.ModelCheckpoint(best_modelname,
                                                        monitor='val_loss', verbose=2,
                                                        save_best_only=True, save_weights_only=True)
    # Finally fit the model to our data
    model.fit(tokens_train, Y_train, verbose=1, epochs=num_epochs,
              callbacks=[earlystopping, chkpt_calllback], class_weight = class_weights,
              batch_size=batchsize, validation_data=(tokens_dev, Y_dev))

    # Print final accuracy for the model (clearer overview)
    test_set_predict(model, tokens_dev, Y_dev, "bestmodel-on-dev", encoder)
    return model


def convert_traindev(max_seq_len, tokenizer, X_train, X_dev):
    # Transform words to indices using a vectorizer
    tokens_train = vectorize_inputtext(max_seq_len, tokenizer, X_train)
    tokens_dev = vectorize_inputtext(max_seq_len, tokenizer, X_dev)
    return tokens_train, tokens_dev


def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()
    print(args)

    # Make reproducible as much as possible
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    python_random.seed(args.seed)

    # Read in the data and embeddings
    X_train, Y_train, X_dev, Y_dev = read_data(args)

    # Create model
    model, tokenizer = create_model(args, Y_train, learning_rate = args.learning_rate, 
                                    lm = args.langmodel_name, batchsize = args.batch_size,
                                    num_epochs = args.num_epochs)
    tokens_train, tokens_dev = convert_traindev(args.max_seq_len, tokenizer, X_train, X_dev)
    encoder, Y_train_bin, Y_dev_bin = numerize_labels(Y_train, Y_dev)

    class_weights = create_class_weights(Y_train, encoder)
    # Train the model
    model = train_model(model, tokens_train, Y_train_bin, tokens_dev,
                        Y_dev_bin, encoder= encoder ,batchsize = args.batch_size,
                        num_epochs = args.num_epochs, class_weights = class_weights,
                        best_modelname = args.output_modelname)

    save_picklefile(encoder, f"{args.output_modelname}.pickle")
    print(f"Label encoder is saved to {args.output_modelname}.pickle")

    details = [args.langmodel_name, args.max_seq_len]
    save_picklefile(details, f"{args.output_modelname}.details")
    print(f"Base language model & max-seq-len are saved to {args.output_modelname}.details")

if __name__ == '__main__':
    main()


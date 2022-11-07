
import argparse

import sys
sys.path.append("../")

from utils import load_picklefile, test_set_predict, write_preds
from lstm_utils import read_testdata_andvectorize, load_vectorizer_pickle
from keras.models import load_model

def create_arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_file", type=str, default='../../../data/test.tsv', required= True,
                        help="If added, use trained model to predict on test set")

    parser.add_argument("--best_modelname", default="models/best_model.h5", type=str,
                        help="Name of the trained model that will be saved after training")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for training")

    parser.add_argument("--output_predfile", type=str, default='preds.txt', required= True,
                        help="File to store the predictions. Each prediction in a line")

    parser.add_argument("--show_cm", default=True, type=bool,
                        help="Show confusion matrix")

    args = parser.parse_args()
    return args

def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()
    print(args)
    encoder = load_picklefile(f"{args.best_modelname}.pickle")
    vectorizer = load_vectorizer_pickle(f"{args.best_modelname}.details.vec")
    # load the saved model
    model = load_model(args.best_modelname)

    X_test, Y_test, tokens_test, Y_test_bin = read_testdata_andvectorize(args.test_file, vectorizer, encoder)
    Y_pred, Y_test = test_set_predict(model, tokens_test, Y_test_bin,
                    "test", encoder, showplot=args.show_cm)
    write_preds(X_test, Y_test, Y_pred, args.output_predfile)



if __name__ == '__main__':
    main()

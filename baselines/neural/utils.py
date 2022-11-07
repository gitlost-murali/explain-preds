
import emoji
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from wordsegment import load, segment
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelBinarizer

import pickle
load()

def space_special_chars(wrd):
    return ''.join(e if (e.isalnum() or e==" ") else f" {e} " for e in wrd)


def segment_hashtags(stg):
    if "#" in stg:
        words = stg.split()
        words = [" ".join(segment(wrd)) if '#' in wrd else wrd for wrd in words]
        stg = " ".join(words)
    return stg

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

def calculate_confusion_matrix(Y_test, y_pred, labels):
    matrix = confusion_matrix(Y_test, y_pred)
    # Convert to pandas dataframe confusion matrix.
    matrix = (pd.DataFrame(matrix, index=labels, columns=labels))
    return matrix


def plot_confusion_matrix(matrix):
    fig, _ = plt.subplots(figsize=(9, 8))
    sn.heatmap(matrix, annot=True, cmap=plt.cm.Blues, fmt='g')
    # show the picture
    plt.show()
    fig.savefig("heatmap.png")
    return

def create_class_weights(Y_train, encoder):
    class_weightscores = class_weight.compute_class_weight(class_weight = 'balanced'
                                                ,classes = encoder.classes_
                                                ,y = Y_train)
    classname2id = dict((name, ix) for (ix, name) in enumerate(encoder.classes_))
    class_weights = dict( (k,v) for (k,v) in zip(encoder.classes_, class_weightscores))
    print(f"Class weights are {class_weights}")
    class_weights = dict( (classname2id[k],v) for (k,v) in class_weights.items())
    print(f"Class weights are {class_weights}")
    return class_weights

def write_preds(X_test, Y_test, Y_pred, filename):
    """Write test predictions along with inputs and expected outputs
    Args:
        X_test (List): Text test sentences
        Y_test (List): Labels of test dataset
        Y_pred (List): Labels predicted
    """
    txtt = []
    for x, yt, yprd in zip(X_test, Y_test, Y_pred):
        txtt.append("\t".join([x,yt,yprd]))

    with open(filename, "w") as fp:
        fp.write("\n".join(txtt))

def read_data(args):
    # Read in the data
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    return X_train, Y_train, X_dev, Y_dev

def test_set_predict(model, X_test, Y_test, ident, encoder, showplot = False):
    '''Do predictions and measure accuracy on our own test set (that we split off train)'''
    # Get predictions using the trained model
    try:
        Y_pred = model.predict(X_test).logits # For BERT
    except:
        Y_pred = model.predict(X_test)
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy

    Y_test = [el[0] for el in list(Y_test)]
    Y_pred = [encoder.classes_[el] for el in Y_pred]
    Y_test = [encoder.classes_[el] for el in Y_test]

    print('Accuracy on own {1} set: {0}'.format(round(accuracy_score(Y_test, Y_pred), 3), ident))
    print('Macro-F1 on own {1} set: {0}'.format(round(f1_score(Y_test, Y_pred, average="macro"), 3), ident))
    if showplot:
        # get the classnames from encoder
        classnames = encoder.classes_
        matrix = calculate_confusion_matrix(Y_test, Y_pred, classnames)
        plot_confusion_matrix(matrix)
    return Y_pred, Y_test

def save_picklefile(inp_object, filename):
    with open(filename, "wb") as fh:
        pickle.dump(inp_object, fh)

def load_picklefile(filename):
    with open(filename, "rb") as fh:
        saved_obj = pickle.load(fh)
    return saved_obj

def numerize_labels(Y_train, Y_dev):
    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)
    return encoder, Y_train_bin, Y_dev_bin

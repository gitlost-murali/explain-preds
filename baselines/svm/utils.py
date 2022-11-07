import pickle

def save2pickle(obj, filename):
    with open(filename, 'wb') as fh:
        pickle.dump(obj, fh)

def load_pickle(filename):
    with open(filename, 'rb') as fh:
        obj = pickle.load(fh)
    return obj

def write_to_textfile(input_string, filename):
    with open(filename, "w", encoding="utf8") as fh:
        fh.write(input_string)
import emoji
from wordsegment import load, segment
load()
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def space_special_chars(wrd):
    return ''.join(e if (e.isalnum() or e==" ") else f" {e} " for e in wrd)

def segment_hashtags(stg):
    if "#" in stg:
        words = stg.split()
        words = [" ".join(segment(wrd.replace("#",''))) if '#' in wrd else wrd for wrd in words]
        stg = " ".join(words)
    return stg


def calculate_confusion_matrix(Y_test, y_pred, labels,):
    matrix = confusion_matrix(Y_test, y_pred)
    # Convert to pandas dataframe confusion matrix.
    print(matrix)
    matrix = (pd.DataFrame(matrix, index=labels, columns=labels))
    return matrix

def plot_confusion_matrix(matrix):
    fig, _ = plt.subplots(figsize=(9, 8))
    sn.heatmap(matrix, annot=True, cmap=plt.cm.Blues, fmt='g')
    # show the picture
    # plt.show()
    fig.savefig("heatmap.png")
    return

def write_to_textfile(input_string, filename):
    with open(filename, "w", encoding="utf8") as fh:
        fh.write(input_string)

def write_preds_tofile(outs, predkeyname, templatehandler, filename):
    verbalizer2label = dict((v,k) for (k,v) in templatehandler.labelmapper.items())
    total_preds = []
    for item in outs:
        batch_preds = item[predkeyname]
        batch_preds = [verbalizer2label.get(prd, "NA") for prd in batch_preds]
        total_preds.extend(batch_preds)

    write_to_textfile("\n".join(total_preds), filename)

def flatten_batchdict(outs):
    flatten_items = dict()
    keys_names = list(outs[0].keys())
    
    for eachkey in keys_names:
        flatten_items[eachkey] = []

    for item in outs:
        for eachkey in keys_names:
            flatten_items[eachkey].extend(item[eachkey])
    return flatten_items

def debug_wo_template(outs, filename):
    incorrect_items = []
    flatten_items = flatten_batchdict(outs)
    for snt, prd, gt in zip(flatten_items['input_sentences'],flatten_items['preds'], flatten_items['gts']):
        if prd != gt:
            incorrect_items.append([snt,prd,gt])
    df = pd.DataFrame(incorrect_items, columns=list(outs[0].keys()))
    df.to_csv(filename)

def debug_w_template(outs, filename):
    incorrect_items = []
    flatten_items = flatten_batchdict(outs)
    for snt, e_prd, e_gt, prd, gt in zip(flatten_items['input_sentences'],
                            flatten_items['preds'], flatten_items['gts'],
                            flatten_items['onlylabel_preds'], flatten_items['onlylabel_gts']):
        if prd != gt:
            incorrect_items.append([snt,prd,gt, e_prd, e_gt])
    df = pd.DataFrame(incorrect_items, columns = list(outs[0].keys()))
    df.to_csv(filename)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np


def plot_confusion_matrix(matrix, labels):
  plt.imshow(matrix, cmap=plt.cm.Greens)
  indices = range(len(matrix))
  plt.xticks(indices, labels)
  plt.yticks(indices, labels)
  plt.colorbar()
  plt.xlabel('Y_pred')
  plt.ylabel('y_test')

  # show the data
  for first_index in range(len(matrix)):
    for second_index in range(len(matrix[first_index])):
      plt.text(first_index, second_index, matrix[first_index][second_index])

  # show the picture
  plt.show()
  return

"""# SCORES"""
def calculate_scores(y_test, Y_pred, showplot = False):
  # Compute precision, recall and F-score for each class, and produce a confusion matrix
  labels = np.unique(y_test)  
  scores_labels = precision_recall_fscore_support(y_test, Y_pred, labels=labels)
  print(pd.DataFrame(scores_labels, columns=labels, index=["Precision", "Recall", "F-score", "Support"]).drop(["Support"]), '\n')
  scores_labels = dict(zip(["Precision", "Recall", "F-score", "Support"], scores_labels))
  macro_fscore = f1_score(y_test, Y_pred, average='macro')

  matrix = confusion_matrix(y_test, Y_pred, labels=labels)
  if showplot:
    # Print confusion matrix.
    print(pd.DataFrame(matrix, index=labels, columns=labels), '\n')
    plot_confusion_matrix(matrix, labels)
    
  return scores_labels, macro_fscore

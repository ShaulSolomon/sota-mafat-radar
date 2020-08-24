import numpy as np
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from matplotlib.colors import LinearSegmentedColormap
import configparser
import matplotlib.patches as patches
import math
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model


def make_tsne(model,data,labels,preds,test,layer_name='dense_1'):
  """
  make TSNE visualization of the data overlayed by labels and misclassifications

  Arguments:
    model -- {keras} -- model variable
    data -- {dictionary} -- datapoints. shape (?,128,32,1)
    labels -- (ndarray) 0/1 for correct labels
    preds -- (ndarray) 0/1 for model predictions
    test -- (ndarray) test datapoints. shape (?,128,32,1). used to highlight their locations 
    layer_name -- (string) layer in the model to extract the predictions from
  """
  intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
  intermediate_output = intermediate_layer_model.predict(data)
  print(intermediate_output.shape)
  tsne_data = intermediate_output

  # possibly append the test data
  if test.shape[0]!=0:
    test_output = intermediate_layer_model.predict(test)
    print(test_output.shape)
    _td = np.concatenate( (tsne_data, test_output))
    tsne_data = _td

  # assign a color for each type of signal
  colors = []
  missc = 0

  if labels.shape[0]==0:
    for i in range(data.shape[0]):
      colors.append('magenta')

  else:
    for i in range(labels.shape[0]):
      color = 'green' if labels[i]==0 else 'blue'
      if labels[i]!=preds[i]:
        color = 'red' #2
        missc += 1
      colors.append(color)
      
  if test.shape[0]!=0:
    for i in range(test.shape[0]):
      colors.append('cyan')

  print("misclassify count=", missc)

  tmodel = TSNE(metric='cosine',perplexity=5, n_iter=1000)
  transformed = tmodel.fit_transform(tsne_data)

  # plot results 

  from matplotlib.pyplot import figure
  figure(figsize=(10,10))
  plt.xticks([])
  plt.yticks([])
  x = transformed[:,0]
  y = transformed[:,1]
  plt.scatter(x, y, c=colors, alpha=.65)
  plt.show()
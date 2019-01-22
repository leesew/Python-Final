import pandas as pd
import numpy as np

from keras import layers, optimizers, regularizers
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.models import Sequential

from keras.utils import plot_model
#from kt_utils import *
import keras.backend as K

import seaborn as sns

from sklearn import preprocessing, model_selection

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

data = pd.read_csv("dataset/winequality-red.csv")
#data["quality"] = data["quality"].astype(object)
print(data.head(5))
print(data.info())
print(data.describe())
g = sns.pairplot(data, vars=["fixed acidity", "volatile acidity","citric acid"], hue="quality")
plt.show(g)

h = sns.pairplot(data, vars=["residual sugar", "chlorides","free sulfur dioxide","total sulfur dioxide"], hue="quality")
plt.show(h)

i = sns.pairplot(data, vars=["density","pH","sulphates","alcohol"], hue="quality")
plt.show(i)

j = sns.countplot(x="quality", data=data)
plt.show(j)

plt.figure(figsize=(13,13))
sns.heatmap(data.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.gist_heat, linecolor='white', annot=True)
plt.show()

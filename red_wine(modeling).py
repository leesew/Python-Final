import pandas as pd
import numpy as np

from keras import layers, optimizers, regularizers
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.models import Sequential
from sklearn import preprocessing, model_selection

from keras.utils import plot_model
#from kt_utils import *
import keras.backend as K

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

data = pd.read_csv("dataset/winequality-red.csv")
data["quality"] =data["quality"].astype(int)
data = pd.get_dummies(data, columns=["quality"])
print(data.head(5))

import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

X = data.iloc[:,0:11].values # first columns
Y = data.iloc[:,12:].values # last columns

X = preprocessing.normalize(X, axis = 0)

#교차 검증을 위한 데이터 분리
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.2, random_state=seed)

#결과 출력
print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)

#모델 구성하기
winemod1 = Sequential()
# layer 1
winemod1.add(Dense(30, input_dim=11, activation='relu', name='fc0',kernel_regularizer=regularizers.l2(0.01)))
winemod1.add(BatchNormalization(momentum=0.99, epsilon=0.001))
#layer 2
winemod1.add(Dense(50, name='fc1',bias_initializer='zeros'))
winemod1.add(BatchNormalization(momentum=0.99, epsilon=0.001))
winemod1.add(Activation('tanh'))
winemod1.add(Dropout(0.5))
#layer 3
winemod1.add(Dense(100, name='fc2',bias_initializer='zeros'))
winemod1.add(BatchNormalization(momentum=0.99, epsilon=0.001))
winemod1.add(Activation('relu'))
winemod1.add(Dropout(0.5))
#layer 4
winemod1.add(Dense(5, name='fc3',bias_initializer='zeros'))
winemod1.add(BatchNormalization(momentum=0.99, epsilon=0.001))
winemod1.add(Activation('softmax'))
winemod1.summary()

#모델링 학습 컴파일
Adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
winemod1.compile(optimizer = Adam, loss = "categorical_crossentropy", metrics = ["categorical_accuracy"])
winemod1.fit(x = X_train, y = Y_train, epochs = 200,verbose=1, batch_size = 64,validation_data=(X_test, Y_test))

preds = winemod1.evaluate(x = X_test, y = Y_test)
print()
print ("Loss = " + str(preds[0]))
print ("예측 정확도 = " + str(preds[1]))

vs.feature_plot(importances, X_train, y_train)
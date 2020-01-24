# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:12:40 2020

@author: Charles
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Activation,LeakyReLU,BatchNormalization,Input
from keras.utils import np_utils
import numpy as np
from keras.regularizers import l2
from keras import backend as K
from keras.layers import Dropout
from keras.optimizers import Adam

#%%
X, y = load_breast_cancer(return_X_y=True)
print(X.shape, y.shape)

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y, train_size = 0.8, random_state = 88)
#%%
#%% Neural Network
K.clear_session()
#%%
model = Sequential()
#model.add(Input(shape=(30,)))
model.add(Dense(240, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(120, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
#%%
adam = Adam(learning_rate = 0.1)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
#%%
history = model.fit(Xtrain, ytrain, validation_split=0.20
                    , epochs=1000, batch_size=4)
#%%
model.summary()
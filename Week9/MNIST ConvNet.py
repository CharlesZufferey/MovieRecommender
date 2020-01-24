# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:46:24 2020

@author: Charles
"""

from keras.models import Sequential
from keras.layers import Dense,Activation,LeakyReLU,BatchNormalization,Conv2D,Flatten
from keras.utils import np_utils,to_categorical
import numpy as np
from keras.regularizers import l2
from keras import backend as K
from keras.layers import Dropout,MaxPooling2D
import matplotlib.pyplot as plt
#%%

from keras.datasets import mnist

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
#%%
xtrain = np.expand_dims(xtrain, axis=3)
xtest = np.expand_dims(xtest, axis=3)
#%%
#xtrain = (xtrain / 255) - 0.5
#xtest = (xtest / 255) - 0.5
ytrain = to_categorical(ytrain)
#%% Conv Neural Network
K.clear_session()

#%%
model = Sequential()

model.add(Conv2D(32,3, padding='same',activation='relu',input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(64,3))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))
#%%
model.compile(loss='categorical_crossentropy'
              , optimizer='adam', metrics=['accuracy'])
#%%
history = model.fit(xtrain, ytrain, validation_split=0.20
                    , epochs=4, batch_size=100)
#%%
nb = 10
#ytrain = np_utils.to_categorical(ytrain, nb)
ytest  = np_utils.to_categorical(ytest, nb)
score = model.evaluate(xtrain, ytrain, batch_size=100)
print(score)
predicted_classes = model.predict_classes(xtest)
correct_indices   = np.nonzero(predicted_classes == ytest.argmax(axis=-1))[0]
incorrect_indices = np.nonzero(predicted_classes != ytest.argmax(axis=-1))[0]
#correct= 9175
#incorect= 825
plt.figure(1, figsize=(7,7))
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(xtest[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], ytest[correct].argmax(axis=-1)))
    plt.xticks([])
    plt.yticks([])
    
plt.figure(2, figsize=(7,7))
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(xtest[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], ytest[incorrect].argmax(axis=-1)))
    plt.xticks([])
    plt.yticks([])
    
    
    
#%%
from matplotlib.pyplot import imread
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import decode_predictions
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
#%%
resnet = ResNet50(weights='imagenet')
#%%
resnet.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
#%%
#a = imread(r'C:\Users\Charles\Documents\Spiced\Week9\fish2.jpg')
#a=a.reshape(1,224,224,3)
#%%
img = image.load_img(r'C:\Users\Charles\Documents\Spiced\Week9\fish.jpg'
                     ,target_size=(224,224))
img = image.img_to_array(img)
o = plt.imshow(img / 255.)
x = preprocess_input(np.expand_dims(img.copy(), axis=0))
preds = resnet.predict(x)
print(decode_predictions(preds, top=5))
#%%

    
    
    
    
    
    
    
    
    
    
    
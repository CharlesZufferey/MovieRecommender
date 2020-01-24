# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:24:01 2020

@author: Charles
"""

from keras.models import Sequential
from keras.layers import Dense,Activation,LeakyReLU,BatchNormalization
from keras.utils import np_utils
import numpy as np
from keras.regularizers import l2
from keras import backend as K
from keras.layers import Dropout
#%%

from keras.datasets import mnist

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
#%%
from matplotlib import pyplot as plt

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(xtrain[i], cmap=plt.cm.Greys)
    plt.axis('off')

#%%
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
#%%
m= RandomForestClassifier ()
#%% Reshaping
X_train = xtrain.reshape(60000, 784)
X_test = xtest.reshape(10000, 784)


#%%
y_train = pd.DataFrame(ytrain)
y_train = pd.get_dummies(y_train,columns=[0])
y_test = pd.DataFrame(ytest)
y_test = pd.get_dummies(y_test,columns=[0])
#%% small RFC test
X_train_1 = X_train[:1000]
X_test_1 = X_test[:1000]
y_train_1 = y_train[:1000]
y_test_1 = y_test[:1000]

#%% Score = 48%
m.fit(X_train_1,y_train_1)
m.predict(X_test_1)
score = m.score(X_test_1,y_test_1)

#%% Neural Network
K.clear_session()
#%%
X_train = xtrain.reshape(60000, 784)
X_train = X_train.astype('float32') 
X_train  /= 255
X_test = xtest.reshape(10000, 784)
X_test = X_test.astype('float32') 
X_test  /= 255
nb = 10
y_train = np_utils.to_categorical(ytrain, nb)
y_test  = np_utils.to_categorical(ytest, nb)
#%% Test1: Accuracy 0.96
model = Sequential([
    Dense(10, input_shape=(784,)),
    Activation('sigmoid'),
    Dense(10),
    Activation('sigmoid'),
])

model.add(Dense(10, activation='relu', input_shape=(784,)
                , W_regularizer=l2(0.001)))
model.add(Dense(10,activation='softmax', input_shape=(784,)))
model.add(Dropout(0.2, input_shape=(784,)))

#%%

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#%%
model.fit(X_train, y_train, epochs=1000, batch_size=500)
#%%
score = model.evaluate(X_train, y_train, batch_size=500)
print(score)
predicted_classes = model.predict_classes(X_test)
correct_indices   = np.nonzero(predicted_classes == y_test.argmax(axis=-1))[0]
incorrect_indices = np.nonzero(predicted_classes != y_test.argmax(axis=-1))[0]
#correct= 9175
#incorect= 825
plt.figure(1, figsize=(7,7))
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct].argmax(axis=-1)))
    plt.xticks([])
    plt.yticks([])
    
plt.figure(2, figsize=(7,7))
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect].argmax(axis=-1)))
    plt.xticks([])
    plt.yticks([])

#%% Test 2 99.2 accuracy
model = Sequential()

model.add(Dense(392, activation='relu', input_shape=(784,)))
                #, W_regularizer=l2(0.001)))
model.add(BatchNormalization(input_shape=(784,)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization(input_shape=(784,)))
model.add(Dropout(0.5, input_shape=(784,)))
model.add(Dense(10,activation='softmax', input_shape=(784,)))


#%%

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#%%
history = model.fit(X_train, y_train, validation_split=0.20, epochs=200, batch_size=1000)
#%%
score2 = model.evaluate(X_train, y_train, batch_size=500)
print(score2)
predicted_classes2 = model.predict_classes(X_test)
correct_indices2   = np.nonzero(predicted_classes2 == y_test.argmax(axis=-1))[0]
incorrect_indices2 = np.nonzero(predicted_classes2 != y_test.argmax(axis=-1))[0]
#correct= 9808
#incorect= 192
plt.figure(1, figsize=(7,7))
for i, correct in enumerate(correct_indices2[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes2[correct], y_test[correct].argmax(axis=-1)))
    plt.xticks([])
    plt.yticks([])
    
plt.figure(2, figsize=(7,7))
for i, incorrect in enumerate(incorrect_indices2[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes2[incorrect], y_test[incorrect].argmax(axis=-1)))
    plt.xticks([])
    plt.yticks([])
    
#%%
#Learning Curve
#history = model.fit(X_train, y_train, validation_split=0.33
#                    , epochs=100, batch_size=1000, verbose=0)
w = model.get_weights()
model.weights

[v.name for v in model.weights]

# plot weights of one layer (for MNIST)
import matplotlib.pyplot as plt

plt.imshow(w[0])#.reshape((784,)))
plt.show()
#%%
model.summary()

#%%
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#%%
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
#%%
l = model.layers
for i in l:
    print(i)
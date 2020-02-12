# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:27:44 2020

@author: Charles
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:59:32 2020

@author: Charles
"""

import os, shutil
#%%
#%%Ideas of layers: going from bigger image shapes to smaller (224 to 112)
#%%224 to 112 to 56 to 28 to 14 to 7 to 1 for pixels
#%% 3 to 64 to 128 to 256 to 512 to 512 to 512 to 4096 to binary
#%%Max pooling before each new convolution
#%%new shaping for each convolution, with ReLU
#%%softmax at the end
#%%Big divider happening for the VGG model> flattening. Oh the left side
#%% matrix of 7 x 7 x 512. After flattening, it is a vector with almost 25088
#%% items. We can describe every image with these 25k features.
#%%divide pixel color by 255 as it might work faster


#%% Main folder
dataset = '/Users/Charles/Documents/Spiced/FinalProject/ImgTest2'
#%% creation of folders for train / test
train_dir = os.path.join(dataset, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(dataset, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(dataset, 'test')
os.mkdir(test_dir)
#%% Creation of sub-folders for the categories
trainprimary = os.path.join(train_dir, 'primary')
#os.mkdir(trainsilo)

trainsecondary = os.path.join(train_dir, 'secondary')
#os.mkdir(trainenviro)

validationprimary = os.path.join(validation_dir, 'primary')
#os.mkdir(validationsilo)

validationsecondary = os.path.join(validation_dir, 'secondary')
#os.mkdir(validationenviro)

testprimary = os.path.join(test_dir, 'primary')
#os.mkdir(testsilo)

testsecondary = os.path.join(test_dir, 'secondary')
#os.mkdir(testenviro)
#%%
from keras import backend as K

K.clear_session()


#%% number of images in dir
print('total images: ',len(os.listdir(trainprimary))) #800
print('total images: ',len(os.listdir(trainsecondary))) #510
print('total images: ',len(os.listdir(validationprimary))) #150
print('total images: ',len(os.listdir(validationsecondary))) #150
print('total images: ',len(os.listdir(testprimary))) #207
print('total images: ',len(os.listdir(testsecondary))) #121

#%% Design and training of the model
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
#%% compile the model
from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

#%% training model with data augmentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)
#train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=50,#100
      epochs=50,#100
      validation_data=validation_generator,
      validation_steps=25)#50
#%%
model.save('/users/charles/documents/spiced/FinalProject/secvsprimary.h5')

#%% Results visualisation
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
#%% test
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
#%% Prediction
img1 = '/Users/Charles/Documents/Spiced/FinalProject/TestModelsImages/Image3.jpg' #primary
img2 = '/Users/Charles/Documents/Spiced/FinalProject/TestModelsImages/Image4.jpg' #primary
img3 = '/Users/Charles/Documents/Spiced/FinalProject/TestModelsImages/Img3.jpg' #secondary
img4 = '/Users/Charles/Documents/Spiced/FinalProject/TestModelsImages/Img4.jpg' #secondary

from keras.preprocessing import image
import numpy as np

img1 = image.load_img(img1, target_size=(150, 150))
img_tensor1 = image.img_to_array(img1)
img_tensor1 = np.expand_dims(img_tensor1, axis=0)
img_tensor1 /= 255.
img2 = image.load_img(img2, target_size=(150, 150))
img_tensor2 = image.img_to_array(img2)
img_tensor2 = np.expand_dims(img_tensor2, axis=0)
img_tensor2 /= 255.
img3 = image.load_img(img3, target_size=(150, 150))
img_tensor3 = image.img_to_array(img3)
img_tensor3 = np.expand_dims(img_tensor3, axis=0)
img_tensor3 /= 255.
img4 = image.load_img(img4, target_size=(150, 150))
img_tensor4 = image.img_to_array(img4)
img_tensor4 = np.expand_dims(img_tensor4, axis=0)
img_tensor4 /= 255.
#%%
print(img_tensor4.shape)
#%%
print(model.predict(img_tensor1))
print(model.predict(img_tensor2))
print(model.predict(img_tensor3))
print(model.predict(img_tensor4))
#%%
print(train_generator.class_indices)
#%%


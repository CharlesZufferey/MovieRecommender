# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:35:12 2020

@author: Charles
Plan: 
    1) run deletefolder script on console
    2) enter URL on flask website
    3) run spider (URL should be coming from flask) on console
    4) run SpiderPrediction from flask (should automatically come up with validate from 2.)

"""


#%%
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
import os, shutil
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
import numpy
import pandas as pd

#%% Loading NN models

m1 = load_model('/Users/Charles/Documents/Spiced/FinalProject/silovsenviro.h5')
m2 = load_model('/Users/Charles/Documents/Spiced/FinalProject/secvsprimary.h5')
m3 = load_model('/Users/Charles/Documents/Spiced/FinalProject/color.h5')
m4 = load_model('/Users/Charles/Documents/Spiced/FinalProject/color2.h5')

#%% using pretrained models to predict type of images
def prediction(path): 
        #for i in range(len(os.listdir(folder))):
        #im = os.listdir(folder)[i]

    im = path
    #im = f'/Users/Charles/Documents/Spiced/FinalProject/FinalProject_Scrapy1/FinalProject_Scrapy1/spiders/images/full/{im}'
    img = image.load_img(im, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    if m1.predict(img_tensor) > 0.55:
        #print("Type: silhouette")
        a = "Silhouette"
    elif m1.predict(img_tensor) < 0.45:
        #print("Type: environmental")
        a = "Environmental"
    else:
        #print ("Type: Unknown")
        a = "Unknown"
    if m2.predict(img_tensor) > 0.50:
        #print("Type: secondary")
        b = "Secondary"
    elif m2.predict(img_tensor) < 0.50:
        #print("Type: primary")
        b = "Primary"
    else:
        #print ("Type: Unknown")
        b = "Unkown"
    colorlist = ['black','blue','red','white','brown','grey','yellow']
    predlist = []
    df = pd.DataFrame()
    for i in range (7):
        prediction2 = str(m3.predict(img_tensor)).replace("[","").replace("   "," ").replace("  "," ").replace("]","").split(" ")[i]
        predlist.append(float(prediction2))
    df = pd.DataFrame(colorlist,predlist).reset_index()
    c = df.loc[(np.argmax(df['index'].values))].values[1]
    return a,b,c
           
#%%
# img = os.listdir('/Users/Charles/Documents/Spiced/FinalProject/flask/static/images/')
# for i in img:
#     path = f'/Users/Charles/Documents/Spiced/FinalProject/flask/static/images/{i}'
#     print (i)
#     type1, type2, color = prediction(path)

#     aa,bb,cc = prediction(path)
#     print (i," ",aa," ",bb," ",cc)
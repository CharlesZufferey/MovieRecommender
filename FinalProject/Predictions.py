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
from keras.models import load_model
import os, shutil
#%% Loading NN models
m1 = load_model('/Users/Charles/Documents/Spiced/FinalProject/silovsenviro.h5')
m2 = load_model('/Users/Charles/Documents/Spiced/FinalProject/secvsprimary.h5')


#%% using pretrained models to predict type of images
def prediction(path): 
    from keras.preprocessing import image
    import numpy as np
    from keras.models import load_model
    import os, shutil
    #%% Loading NN models
    m1 = load_model('/Users/Charles/Documents/Spiced/FinalProject/silovsenviro.h5')
    m2 = load_model('/Users/Charles/Documents/Spiced/FinalProject/secvsprimary.h5')
        #for i in range(len(os.listdir(folder))):
        #im = os.listdir(folder)[i]
    im = path
    print(im)
    #im = f'/Users/Charles/Documents/Spiced/FinalProject/FinalProject_Scrapy1/FinalProject_Scrapy1/spiders/images/full/{im}'
    img = image.load_img(im, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    if m1.predict(img_tensor) > 0.75:
        #print("Type: silhouette")
        a = "silhouette"
    elif m1.predict(img_tensor) < 0.25:
        #print("Type: environmental")
        a = "environmental"
    else:
        #print ("Type: Unknown")
        a = "Unknown"
    if m2.predict(img_tensor) > 0.55:
        #print("Type: secondary")
        b = "Secondary"
    elif m2.predict(img_tensor) < 0.45:
        #print("Type: primary")
        b = "Primary"
    else:
        #print ("Type: Unknown")
        b = "Unkown"
    return a,b
           

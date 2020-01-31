# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 09:26:44 2020

@author: Charles
"""
from sqlalchemy import create_engine
import pandas as pd
import pickle
import numpy as np
from sklearn.decomposition import NMF
import pandas as pd
import random
#%%
conns = 'postgres://postgres:postgres@localhost/MovieRec'
#%%
db = create_engine(conns, encoding='latin1', echo=False)
#%% reviews for movies having between 50 and 70 reviews for first 10k users
spm = """select * from review"""
df_spm = pd.read_sql(spm,db)
#%%
a = df_spm[['userid','title','avgrating']]

#%%
a.set_index(['userid','title'],inplace=True)

#%%
spm1 = a.unstack()

#%%
spm1 = spm1.fillna(2)

# #%%

# #create a model and set the hyperparameters
# # model assumes R ~ PQ'
# model = NMF(n_components=20, init='random', random_state=10)
# #%%
# model.fit(spm1)
#%%
with open('C://Users/Charles/Documents/Spiced/week10/flask/nmf_model.bin', 'rb') as f:
    nmf = pickle.load(f)
#%%
Q = nmf.components_  # movie-genre matrix

P = nmf.transform(spm1)  # user-genre matrix

#print(model.reconstruction_err_) #reconstruction error
#%%c1
nR = np.dot(P, Q)
#print(nR) ## The reconstructed matrix!
#%%
import GenreRecommendation as gr
# predict the hidden features for a new data point
#%%
def TopRecommend(m1,m2,m3
                 ,m4,m5,m6
                 ,r1,r2,r3
                 ,r4,r5,r6,n):
    """ function to calculate recommendations """
    selection = []
    top = []
    bottom = []
    rating1 = [m1,r1] #Comedy
    rating2 = [m2,r2] #drama
    rating3 = [m3,r3] #child
    rating4 = [m4,r4] #Comedy
    rating5 = [m5,r5] #drama
    rating6 = [m6,r6]
    selection = [m1,m2,m3,m4,m5,m6]
    top = [m1,m2,m3]
    bottom = [m4,m5,m6]
    randomindividual = []
    resultgenre = []
    for i in range(len(spm1.columns)):
        if spm1.columns[i][1] == rating1[0]:
           randomindividual.append(rating1[1])
        elif spm1.columns[i][1] == rating2[0]:
           randomindividual.append(rating2[1])
        elif spm1.columns[i][1] == rating3[0]:
           randomindividual.append(rating3[1])
        elif spm1.columns[i][1] == rating4[0]:
            randomindividual.append(rating4[1])
        elif spm1.columns[i][1] == rating5[0]:
            randomindividual.append(rating5[1])
        elif spm1.columns[i][1] == rating6[0]:
            randomindividual.append(rating6[1])
        else:
            randomindividual.append(1)
           
           
    #print(rating2)
    randomindividual = pd.DataFrame(randomindividual).transpose()
    
    
    P1 = nmf.transform(randomindividual)
    nR1 = np.dot(P1, Q)
    movie = []
    probarating = []
    for i in range(len(spm1.columns)):
        if spm1.columns[i][1] in selection:
            pass
        else:
            movie.append(spm1.columns[i][1])
            probarating.append(nR1[0][i])
    FinalRecMatrix = pd.DataFrame()
    FinalRecMatrix['Movie']=movie
    FinalRecMatrix['probability of rating'] = probarating
    FinalRecMatrix = FinalRecMatrix.sort_values(by=['probability of rating'],ascending=False)
       
    top = gr.genrerec(top)
    bottom = gr.genrerec(bottom)
    result_list = []
    for i in range(n):
        result_list.append(FinalRecMatrix.iloc[i][0])
    resultgenre = gr.genrerec2(result_list)
    return result_list,resultgenre,top,bottom
#%%
# resl,resg,t,b = TopRecommend('How Green Was My Valley (1941)',
#               'To Have and Have Not (1944)',
#               'Judgment at Nuremberg (1961)',
#               'Alvin and the Chipmunks (2007)',
#               'Cinderella (2015)',
#               'Teenage Mutant Ninja Turtles (2014)',1,1,1,5,5,5,100)

#%%
# import os
# print(os.getcwd())
# #%%
# os.chdir('C://Users/Charles/Documents/Spiced/week10/flask')
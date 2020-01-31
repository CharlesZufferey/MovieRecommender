# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:10:36 2020

@author: Charles
"""

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

#%%

#create a model and set the hyperparameters
# model assumes R ~ PQ'
model = NMF(n_components=20, init='random', random_state=10)
#%%
model.fit(spm1)
#%%
binary = pickle.dumps(model)
open('C://Users/Charles/Documents/Spiced/week10/flask/nmf_model.bin', 'wb').write(binary)
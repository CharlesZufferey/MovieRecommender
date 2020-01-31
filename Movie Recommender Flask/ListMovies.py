# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:43:43 2020

@author: Charles
"""

from sqlalchemy import create_engine
import pandas as pd
#%%
conns = 'postgres://postgres:postgres@localhost/MovieRec'
#%%
def listmovie():
    db = create_engine(conns, encoding='latin1', echo=False)
    lm = """select * from allmoviesimple"""
    df = pd.read_sql(lm,db)
    list = []
    for i in range(len(df)):
        list.append(df.loc[i][0])
    return list  

#%%

def infos():
    db = create_engine(conns, encoding='latin1', echo=False)
    lm = """select * from allmovies"""
    df = pd.read_sql(lm,db)
    return df
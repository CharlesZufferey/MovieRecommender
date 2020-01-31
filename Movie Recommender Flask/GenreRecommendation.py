# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:20:17 2020

@author: Charles
"""

from sqlalchemy import create_engine
import pandas as pd
#%%
conns = 'postgres://postgres:postgres@localhost/MovieRec'
#%%
def genrerec(list1):
    ls = []
    db = create_engine(conns, encoding='latin1', echo=False)
    mg = """select distinct title, 
        case when position('|' in genres) =0 then genres else left(genres,position('|' in genres)-1) end as genre from movies
        """
    df = pd.read_sql(mg,db)
    df1 =  df[df['title'].isin(list1)]
    for i in df1['genre']:
        ls.append(i)
    ls = list(dict.fromkeys(ls))
    return ls
#%%
def genrerec2(list1):
    ls = []
    db = create_engine(conns, encoding='latin1', echo=False)
    mg = """select distinct title, 
        case when position('|' in genres) =0 then genres else left(genres,position('|' in genres)-1) end as genre from movies
        """
    df = pd.read_sql(mg,db)
    df1 =  df[df['title'].isin(list1)]
    for i in df1['genre']:
        ls.append(i)
    #ls = list(dict.fromkeys(ls))
    return ls
#%%

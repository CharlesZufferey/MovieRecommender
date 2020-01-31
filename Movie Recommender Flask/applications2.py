# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:06:16 2020

@author: Charles
"""
from flask import Flask
from flask import render_template,request,url_for
import ListMovies as lm
from sqlalchemy import create_engine
import pandas as pd
#%%
app = Flask(__name__)

if __name__ =='__main__':
    app.run(host='0.0.0.0',port=80)
    
#%%    
@app.route('/movie')
def movie():
    list1 = lm.listmovie()
    return render_template('dataset2.html',result_html1 = list1)

@app.route('/movie/info', methods=['GET'])
#%%
def info():
    conns = 'postgres://postgres:postgres@localhost/MovieRec'
    db = create_engine(conns, encoding='latin1', echo=False)
    lm2 = """select title,genres,r.movieid,avg(rating) avgrating,count(rating) countrating from movies m
        join ratings r on r.movieid = m.movieid where title in (
        select title from ratings r
        join movies m on m.movieid = r.movieid
        where userid < 10000
        group by title
        having count(rating) > 50 and count(rating)<70)
        group by title,r.movieid,genres """
    df2 = pd.read_sql(lm2,db)
    data = dict(request.args)
    movie = data['movieinfo']
    moviegenre = df2[df2['title']==movie]['genres'].values
    countrating = df2[df2['title']==movie]['countrating'].values
    avgrating = df2[df2['title']==movie]['avgrating'].values
    return render_template('infos.html',moviegenre = moviegenre,countrating = countrating
                           ,avgrating = avgrating,movie = movie)

#%%
# a = info()
# #%%
# o = (a[a['title']=="2 Guns (2013)"][['genres','avgrating','countrating']].values)
# for i in o:
#     print (i)
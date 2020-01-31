from flask import Flask
from flask import render_template,request,url_for
import MovieRec1 as sr
import ListMovies as lm
from sqlalchemy import create_engine
import pandas as pd
#%%
app = Flask(__name__)

if __name__ =='__main__':
    app.run(host='0.0.0.0',port=80)

@app.route('/')
# def hello():
#     return render_template('index.html')

# @app.route('/test')
def test():
    list = lm.listmovie()
    return render_template('dataset.html',result_html = list)

@app.route('/recommendation', methods=['GET'])
def recommend():

    #result = sr.recommend(8) ###FOR NOW OUR INPUT IS HARDCODED!
    #STEP1: Collect the payload
    #returns a dictionary where the keys are the HTML names,
    #and the values are the user input from the form.
    data = dict(request.args)

    movie1 = data['movie1']
    movie2 = data['movie2']
    movie3 = data['movie3']
    movie4 = data['movie4']
    movie5 = data['movie5']
    movie6 = data['movie6']

    #return render_template('recommendation.html', result_html=result)
    # rating1 = float(data['rating1'])
    # rating2 = float(data['rating2'])
    # rating3 = float(data['rating3'])
    rating1 = 5
    rating2 = 5
    rating3 = 5
    rating4 = 0
    rating5 = 0
    rating6 = 0

    num = int(data['num'])

    ###WARNING: Check data types / validate inputs!

    result,Rgenre,top,bottom = sr.TopRecommend(movie1,movie2,movie3
                             ,movie4,movie5,movie6
                             ,rating1,rating2,rating3
                             ,rating4,rating5,rating6
                             ,num) 

    output = result #set(zip(result,Rgenre))
    return render_template('recommendation.html', result_html=output, num=num
                           , top=top,bottom=bottom)

#%%
@app.route('/movie/info', methods=['GET'])
def info():
    df2 = lm.infos()
    data1 = dict(request.args)
    movie1 = data1['movieinfo']
    moviegenre = df2[df2['title']==movie1]['genres'].values
    countrating = df2[df2['title']==movie1]['countrating'].values
    avgrating = df2[df2['title']==movie1]['avgrating'].values
    return render_template('infos.html',moviegenre = moviegenre,countrating = countrating
                           ,avgrating = avgrating,movie = movie1)
  
# Next step: add a "Leaderboard" on the main screen

@app.route('/lb')
#%%
def lb():
    nb = []
    nb1 = []
    conns = 'postgres://postgres:postgres@localhost/MovieRec'
    db = create_engine(conns, encoding='latin1', echo=False)
    lm = """select * from allratings a order by countrating desc"""
    lm1 = """select * from allratings a order by avgrating desc"""
    df = pd.read_sql(lm,db)
    df1 = pd.read_sql(lm1,db)
    movieN = df['title']
    number = df['countrating']
    movieB = df1['title']
    best = df1['avgrating']
    for i in range(len(movieN)):
        nb.append(movieN[i] + " - Nb of reviews: " + str(number[i]))
        nb1.append(movieB[i] + " - Average Rating: " + str(best[i]))
    #return po
    return render_template('lb.html',movieN = nb,movieB = nb1)
#trie = lb()

# alvin and chipmunks
    # cinderella
# madagascar
    
#abraham lincoln
    # amytiville
# centipede
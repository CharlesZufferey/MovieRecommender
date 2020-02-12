from flask import Flask, session
from flask import render_template,request,url_for
import pandas as pd
import os,shutil
import Predictions as pr
import subprocess

"""
Next step: try to train a model for good / bad images. Need to figure out how to find bad images.
Or wrong color.
    set FLASK_APP=applications.py
 flask run --port=80
"""
#%% creation of the flask connection
app = Flask(__name__)
app.secret_key = 'dljsaklqk24e21cjn!Ew@@dsa5'
if __name__ =='__main__':
    app.run(host='0.0.0.0',port=80,threaded=False)

#%%
@app.route('/')
def home():
    return render_template('homepage.html')

#%%
@app.route('/scraping',methods = ['GET'])
def scraping():
    data = dict(request.args)
    url2 = data['url']
    colors = ['black','blue','red','white','brown','grey','yellow']
    session['cs'] = ""
    for i in colors:
        if url2.find(i) > -1:  
            session["cs"] = i
            break
        else:
            1==1
    #colorsearch = session.get("cs",None)
    url = "https://www.wayfair.com/keyword.php?keyword=" + url2.replace(" ", "+")
    cmd_spider = f'cmd.exe /K scrapy runspider -o output.csv -L WARNING SpiderPrediction.py -a start_url="{url}"'
    os.chdir('/Users/Charles/Documents/Spiced/FinalProject/FinalProject_Scrapy1/FinalProject_Scrapy1/spiders')
    cmd = subprocess.Popen(cmd_spider) 
    os.chdir ('/Users/Charles/Documents/Spiced/FinalProject/flask')
    return render_template('scraping.html',url = cmd_spider)

#%%
@app.route('/Status',methods = ['Get'])
def Status():
    nbimdl = len(os.listdir('/Users/Charles/Documents/Spiced/FinalProject/flask/static/images/'))
    full = os.listdir('/Users/Charles/Documents/Spiced/FinalProject/FinalProject_Scrapy1/FinalProject_Scrapy1/spiders/images/Full')
    img = os.listdir('/Users/Charles/Documents/Spiced/FinalProject/flask/static/images/')
    for file in full:
        fullpath = f'/Users/Charles/Documents/Spiced/FinalProject/FinalProject_Scrapy1/FinalProject_Scrapy1/spiders/images/Full/{file}'
        shutil.move(fullpath, '/Users/Charles/Documents/Spiced/FinalProject/flask/static/images/')
    return render_template('inprogress.html',nb = nbimdl)

#%%
@app.route('/images',methods = ['GET'])
def imdisplay():
    img = os.listdir('/Users/Charles/Documents/Spiced/FinalProject/flask/static/images/')
    images = []
    Type = []
    cs = []
    Wrongcol1 = []
    Wrongcol2 = []
    for i in img:
        path = f'/Users/Charles/Documents/Spiced/FinalProject/flask/static/images/{i}'
        type1, type2, color = pr.prediction(path)
        
        images.append(i)
    #return images,Type1,Type2,Type0
        Type0 = type1 + ", " + type2 + ", " + color
        Type.append(Type0)
        colorsearch = session.get("cs",None)
        if colorsearch == color or colorsearch == "":
             cs.append(1)
             if type1 == "Environmental" or type2 == "Secondary": 
                 Wrongcol1.append(i)
                 Wrongcol2.append(Type0)     
        else:
            cs.append(0)
            Wrongcol1.append(i)
            Wrongcol2.append(Type0)
        session['wrongcol1'] = Wrongcol1
        session['wrongcol2'] = Wrongcol2
    return render_template('ImDisplay.html',img = zip(images,Type,cs))
#%%
@app.route('/imagesfiltered',methods = ['GET'])
def imfiltered():
    Wrongcol1 = session.get("wrongcol1",None)
    Wrongcol2 = session.get("wrongcol2",None)
    return render_template('ImFiltered.html',img = zip(Wrongcol1,Wrongcol2))
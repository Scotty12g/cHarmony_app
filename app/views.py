from flask import Flask, request, render_template, url_for
from color_model import FindColor
from app import app
import os
from os.path import join

UPLOAD_FOLDER = 'app/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
STATIC_FOLDER = 'app/static'
app.config['STATIC_FOLDER'] = STATIC_FOLDER

@app.route('/', methods = ['GET','POST'])
def index():
    if request.method=='POST':
        return (url_for('results'))
    return render_template('index.html')

@app.route('/results', methods=['GET','POST'])
def results():
    matchtype = request.form['matchtype']
    tomatch_path = os.path.join(app.config['UPLOAD_FOLDER'],'tomatch.png')
    tomatch=request.files['tomatch']
    tomatch.save(os.path.join(app.config['UPLOAD_FOLDER'],'tomatch.png'))
    closet_path = os.path.join(app.config['UPLOAD_FOLDER'],'closet.png')
    closet=request.files['closet']
    closet.save(os.path.join(app.config['UPLOAD_FOLDER'],'closet.png'))
    save_path=os.path.join(app.config['STATIC_FOLDER'],'final.png')
    result, input, img=FindColor(matchtype, tomatch_path, closet_path)
    img.save(os.path.join(app.config['STATIC_FOLDER'],'final.png'))
    return render_template('results.html',result=result, matchtype=matchtype, input=input,save_path=save_path)


@app.route('/about')
def about():
    return render_template('about.html')
    
@app.route('/results2')
def results2():
    return render_template('results.html')
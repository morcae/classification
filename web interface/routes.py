from bottle import route, view, Bottle, request
from datetime import datetime
from bottle.ext.sqlite import Plugin
import sqlite3 as sql
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as mpld3
import plotly.plotly as py
import plotly.tools as tls
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import os
import random
import pandas as pd

@route('/')
@route('/home')
@view('index')
def home():
    """Renders the home page."""
    return dict(
        year=datetime.now().year
    )

@route('/contact')
@view('contact')
def contact():
    """Renders the contact page."""
    return dict(
        title='Contact',
        message='Your contact page.',
        year=datetime.now().year
    )

@route('/about')
@view('about')
def about():
    """Renders the about page."""
    return dict(
        title='About',
        message='Your application description page.',
        year=datetime.now().year
    )

@route('/second',method=['GET', 'POST'],)
@view('second')
def second():

    # write expert's chouce to DB
    opt = request.forms.get('genre')
    con = sql.connect('classification.db')
    cur = con.cursor()
    filename  = audio.split('.')[0]
    n = cur.execute('''SELECT experts.%s from experts where id = "%s"''' %(opt, filename) )
    n =  n.fetchone()[0] + 1
    cur.execute('''UPDATE experts SET %s = %s WHERE id  = "%s"''' %(opt, n, filename) )
    con.commit()

    # get all scores for filename
    scores = cur.execute('''SELECT * from experts where id = "%s"''' %filename ).fetchall()
    y = scores[0][2:]
    
    # make plot with scores
    py.sign_in("morcae", "mATYBYmeS9FS6jRNXyrz")
    x = [1, 2, 3, 4]

    labels = ['blues', 'classical', 'pop', 'rock']
    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)

    plt.bar(x, y, color="blue", align='center')
    ax.set_ylabel('Scores')
    ax.set_xlabel('blues                classical                 pop                   rock')
    ax.set_title('Scores by experts and genres')
    ax.set_xticks([])
    plot_url = py.plot_mpl(mpl_fig, filename='stacked-bar-chart', auto_open=False)
    plot_url += ".embed"
    print plot_url
    
    con.close()

    return dict(
        title='Second',
        message='Your application description page.',
        year=datetime.now().year, plot_url = plot_url
    )

@route('/player', method='POST')
@view('player')
def player():
    path = "/static/experts/"
    # get filename with low scores from db
    con = sql.connect('classification.db')
    cur = con.cursor()
    id = cur.execute('''SELECT id, experts.blues+experts.classical+experts.pop+experts.rock AS total  FROM experts GROUP BY id ORDER BY total ASC LIMIT 1'''  )
    id = id.fetchone()[0]

    # select genres for file
    filename = cur.execute('''select experts.filename from experts where id = %s''' %id )
    filename = filename.fetchone()[0]
    b = cur.execute('''select ml_classification.blues from ml_classification where filename = "%s"''' %filename )
    b = b.fetchone()[0]
    c = cur.execute('''select ml_classification.classical from ml_classification where filename = "%s"''' %filename )
    c = c.fetchone()[0]
    p = cur.execute('''select ml_classification.pop from ml_classification where filename = "%s"''' %filename )
    p = p.fetchone()[0]
    r = cur.execute('''select ml_classification.rock from ml_classification where filename = "%s"''' %filename )
    r = r.fetchone()[0]
    if b<c and b<p and b<r:
        genres = ['classical', 'pop', 'rock']
    elif c<b and c<p and c<r:
        genres = ['blues', 'pop', 'rock']
    elif p<b and p<c and p<r:
        genres = ['blues', 'classical', 'rock']
    else:
        genres = ['blues', 'classical', 'pop']
    first = genres[0]
    second = genres[1]
    third = genres[2]
    con.close()

    global audio
    audio = str(id) + ".wav"
    path_audio = path + audio
        
    return dict(
        title='Start',
        message='Your application description page.',
        year=datetime.now().year, path_audio = path_audio, first = first, second = second, third = third
    )

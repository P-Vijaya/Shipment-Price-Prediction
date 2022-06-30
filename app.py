from flask import Flask,render_template,request
from flask_cors import cross_origin
#from Preprocessing import Preprocessor
#import matplotlib.pyplot as plt
#import os
#import numpy as np

app = Flask(__name__)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return "Hi"
    #return render_template('index.html')

@app.route("/test",methods=['POST'])
def testForecast():
    try:
        print("Hi")
        #return render_template('results.html', results=results)
    except Exception as e:
        raise Exception(f"(app.py) - Something went wrong"+str(e))


@app.route("/single",methods=['POST'])
def singleForecast():
    try:
        print("Hi")
        #return render_template('results.html', results=results)
    except Exception as e:
        raise Exception(f"(app.py) - Something went wrong"+str(e))

if __name__ == '__main__':
    app.run(port=7000,debug=True)
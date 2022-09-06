# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 10:28:14 2022

@author: amina
"""

#!pip install flask
import pickle
import numpy as np
import requests
import json
from flask import Flask, request

model_nlp_gbc_pk_1 = pickle.load(open("nlp_diff_gbc_new.pkl", "rb"))

app=Flask(__name__)

@app.route('/api_prediction', methods=["GET","POST"])
def api_prediction():
    if request.method=="GET":
        return "Please send Post Request"
    elif request.method=="POST":
        data=request.get_json()

        X = data['Text']
        labels = model_nlp_gbc_pk_1.predict(X).tolist()
        return json.dumps({'predictions': labels})
    
if __name__ == "__main__":
    app.run()
    """
url="http://127.0.0.1:5000/api_prediction"       

data={"Text": " I bought foods for the children and some cutlass for farm works"}
        
r=requests.post(url,json=data)        

print(r)
print(r.text)
"""
from flask import Flask,jsonify,Response,request
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
import urllib.request
from urllib.request import Request

from countface import countF
from trainModel import trainModel
from recognizeFace import recognizeF


import json
import re
from urllib.parse import urlparse
import sys
import os



app=Flask(__name__)
CORS(app)

basePath = os.getcwd() + '/'

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/debug')
def debuggincode():
    currdir = os.getcwd()
    allfiles = os.listdir()
    return {'current Directory' : f"{currdir}", 'directories': f"{allfiles}"}

@app.route('/getCount', methods=["POST"])
def getcount():
    l = request.get_json()
    if 'link' not in l.keys():
        return Response(json.dumps({'count' : 'Error'}), mimetype='application/json', status=404)
    link = l['link']
    print(link)
    print(countF(link))
    return Response(json.dumps({'count': countF(link)}), mimetype='application/json', status=200)

@app.route('/addFace', methods=["POST"])
def addFace():
    data = request.get_json()
    keys = data.keys()
    if 'links' not in keys or 'id' not in keys  or 'doctor' not in keys: 
        return Response(json.dumps({'staus' : 'Error'}), mimetype='application/json', status=404)
    links = data['links']
    id = str(data['id'])
    doctor = int(data['doctor'])
    faceCascade = cv2.CascadeClassifier(basePath + 'haarcascade_frontalface_default.xml')

    if doctor == 1:
        os.makedirs(basePath + 'db/doctor/' + id)
        path = basePath +  'db/doctor/' + id + '/'
    else:
        os.makedirs(basePath + 'db/patient/' + id)
        path = basePath + 'db/patient/' + id + '/'

    # added images
    print('adding images')
    count = 0
    for link in links:
        req = Request(url=link, headers={'User-Agent': 'Mozilla/5.0'})
        req = urllib.request.urlopen(req)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(20, 20)
        )
        if len(faces) > 1:
            continue;
        elif len(faces) == 1:
            for (x,y,w,h) in faces:
                cv2.imwrite(f"{path}{count}.jpg",gray[y:y+h, x:x+w])
        count+=1
    print('training model')
    if doctor == 1:
        done = trainModel(basePath + 'db/doctor/',doctor)
    else:
        done = trainModel(basePath + 'db/patient/', doctor)
    print('trained model')
    if not done:
        return Response(json.dumps({'staus' : 'Error1'}), mimetype='application/json', status=404)
    return Response(json.dumps({'staus' : 'Done'}), mimetype='application/json', status=200)
    


@app.route('/recognizeFace', methods=["POST"])
def recgFace():
    data = request.get_json()
    keys = data.keys()
    if 'link' not in keys or 'doctor' not in keys:
        return Response(json.dumps({'Error': 'Value Missing'}), mimetype='application/json', status=404)
    ans = recognizeF(data['link'], int(data['doctor']))
    return Response(json.dumps({'found': ans}), mimetype='application/json', status=200)



if __name__ == '__main__':
    from waitress import serve
    server(app,host='0.0.0.0', port=8000)
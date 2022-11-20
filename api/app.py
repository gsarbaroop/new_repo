from flask import Flask
from flask import request
from joblib import load
import os

app = Flask(__name__)
model_path = r"./models/svm_gamma=0.001_C=0.2.joblib"
model = load(model_path)

@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"


# get x and y somehow    
#     - query parameter
#     - get call / methods
#     - post call / methods ** 

@app.route("/sum", methods=['POST'])
def sum():
    x = request.json['x']
    y = request.json['y']
    z = x + y 
    return {'sum':z}


@app.route("/predict", methods=['POST'])
def predict_digit():
    lt=[]
    if 'model_name' not in request.json:
        for file in os.listdir(r"./results"):
            f = open(r"./results/"+file,"r")
            for st in f.read():
                lt.append(st)
        return {"file":lt}
    else:
        model_name = request.json['model_name']
        image = request.json['image']
        print("Predicting image")
        predicted = model.predict([image])
        return {"Number predicted":int(predicted[0])}
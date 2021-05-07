Predicting diabetes app with FLASK


```py
from flask import Flask, request
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

def model_predict(value):
  return model.predict(value)

@app.route('/predict')
def predictions():
  value = request.args.get('value')
  prediction = model_predict(np.arange(int(value), 11).reshape(1,-1))
  return f'the result is {prediction}!'

if __name__ == '__main__':
    		app.run('localhost', 5000)

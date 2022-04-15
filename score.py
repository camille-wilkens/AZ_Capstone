import os
import numpy as np

import joblib
import pandas as pd

import json
import sys
import subprocess 
from azureml.core.model import Model


def init():   
    global model
    heart_model = Model.get_model_path(model_name=Heart_Failure_Prediction_Model')
    model = joblib.load(heart_model)


def run(data):
    try:
        data = pd.DataFrame(json.loads(data)['data'])
        result = model.predict(data)                           
        return result.tolist()
                                       
    except Exception as e:
        return json.dumps({"error":str(e)})
       
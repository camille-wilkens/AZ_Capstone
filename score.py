import os
import pickle
import numpy as np

import joblib
import pandas as pd

import json
import sys
import subprocess 
from azureml.core.model import Model


def init():   
    global model
    model_path = Model.get_model_path('hyperdrive_model')
    model = joblib.load(model_path)


def run(data):
    try:
                            
        data = pd.DataFrame(json.loads(data)['data'])
        result = model.predict(data)                           
        return result.tolist()
                                       
    except Exception as e:
        error = str(e)
        return error

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
    model_path = Model.get_model_path(model_name = 'HyperDrive_model')
    model = joblib.load(model_path)


def run(input_data):
    try:
        data = json.loads(input_data)['data']
        data = np.array(data)
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        # return error message back to the client
        return json.dumps({"error": result})


{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import joblib\n",
    "import pandas as pd\n",
    "import os\n",
    "import numpy as np\n",
    "import json\n",
    "import sys\n",
    "import subprocess \n",
    "from azureml.automl.core\n",
    "from azureml.core.model import Model\n",
    "\n",
    "\n",
    "\n",
    "def init():\n",
    "    global model\n",
    "    model_path = Model.get_model_path(model_name='Heart_Failure_Prediction_Model')\n",
    "    model = joblib.load(model_path)\n",
    "\n",
    "\n",
    "def run(data):\n",
    "    try:\n",
    "        #data = pd.DataFrame(json.loads(data)['data'])\n",
    "        result = model.predict(data)\n",
    "        return result.tolist()\n",
    "    except Exception as err:\n",
    "        print(str(err))\n",
    "        return str(err)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

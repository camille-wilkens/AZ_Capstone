
from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#https://knowledge.udacity.com/questions/800388 
# No pandas found error, this KB resolved the issue

import sys
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
import pandas as pd

from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    x_df.drop_duplicates(inplace=True)
 
    x_df.drop(['PassengerId','Cabin','Name','Ticket','Embarked', 'SibSp', 'Parch'],axis=1,inplace=True)
    x_df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    x_df['Age'].fillna(x_df[Age].mean(), inplace=True)
    x_df['Fare'].fillna(x_df[Fare].median(), inplace=True)

    
    y_df = x_df.pop("Survived")
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    # TODO: Create TabularDataset using TabularDatasetFactory
    # Data is located at:
    # "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

    data_url = "https://raw.githubusercontent.com/camille-wilkens/AZ_Capstone/main/Titanic-Dataset.csv"
    ds = TabularDatasetFactory.from_delimited_files(data_url)

    x, y = clean_data(ds)

    # TODO: Split data into train and test sets.

    ### YOUR CODE HERE ###a
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .20, random_state = 10)

    model = LogisticRegression(C=args.C,max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
 #   os.makedirs('outputs',exist_ok=True)
 #   joblib.dump(value=model, filename="model.joblib")

if __name__ == '__main__':
    main()





An overview of the project
An overview of the dataset used
An overview of the method used to get the data into your Azure ML Studio workspace.
An overview of your AutoML experiment settings and configuration
An overview of the types of parameters and their ranges used for the hyperparameter search
An overview of the two models with the best parameters
An overview of the deployed model and instructions on how to query the endpoint with a sample input
A short overview of how to improve the project in the future
ALL the screenshots required with a short description
A link to the screencast video on YouTube (or a similar alternative streaming ser


*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Your Project Title Here

*TODO:* Write a short introduction to your project.

In this Udacity Azure Machine Learning Nanodegree Capstone project, I created two models: one using Automated ML (denoted as AutoML from now on) and one customized model whose hyperparameters are tuned using HyperDrive. I then compared the performance of both the models and deployed the best performing model.  This project used an [external datasets](https://www.kaggle.com/code/yasserh/titanic-survival-predition-top-ml-algorithms/data) and trained a model using different tools available in the AzureML framework as well as deployinging the model as a web service.

## Project Workflow
![project_workflow](./images/workflow.PNG)


### Project Main Steps:

1. [Project Set Up and Installation](#setup)
2. [Dataset](#dataset)
3. [Workspace Access](#access)
4. [Automated ML](#automl)
5. [Hyperparameter Tuning](#hyper)
6. [Consume model endpoints](#endpoints)
7. [Create and publish a pipeline](#pipeline)
8. [Documentation Video](#video)
9. [Future Improvements](#future)


## Project Set Up and Installation

*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

#### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

## Access
*TODO*: Explain how you are accessing the data in your workspace. An overview of the method used to get the data into your Azure ML Studio workspace.

## Automated ML
#### Overview `automl` settings and configuration utlized in this experiment  
#### Settings


#### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

#### Screenshots
Remeber to provide screenshots of the `RunDetails` widget best model trained with it's parameters.

## Hyperparameter Tuning

#### Dataset


#### RandomParamterSampling
Overview of the types of parameters and their ranges used for the hyperparameter search
* 
'''
    {"--C":choice(0.5,1.0),     
    "--max_iter":choice(50,100,150)})  
'''

* I used RandomParameterSampling as it supports continous and discrete hyperparamters.  Another key benefit of using RandomParameterSampling, is its less resource intensive and time consuming and supports early termination.

* The Early Stopping policy, I utilized was the Bandit Policy is also less resource intensive and time consuming.  If a run's performance is outside the best run's slack_factor, the run is early terminated -- saving time and resources.

* 


#### Model Used & Why 
#### Prepare Data
* Download the dataset [Data](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) and convert into   TabularDatasetFactory dataset.
* Clean the dataset (clean_data located in train.py)
* Split data into training and test sets (80/20)
* Utilize a Scikit-learn Logistic Regression Model for Classification
   
#### Hyperparamater Tuning
hyperdrive_config = HyperDriveConfig (
hyperparameter_sampling=RandomParameterSampling(
{"--C":choice(0.5,1.0), --max_iter":choice(50,100,150)})  ,
primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
primary_metric_name='Accuracy',
policy=BanditPolicy(evaluation_interval=1, slack_factor=0.1),
max_total_runs=8, 
max_concurrent_runs=4,
estimator=SKLearn(source_directory='.', entry_script='train.py', compute_target=cluster_name))
           
#### Classifcation Algorithim
* Logistic Regression
   

**Benefits of the parameter sampler chosen:**
  RandomParameterSampling supports continous and discrete hyperparamters.  It is also less resource intensive and time consuming.

**Benefits of the early stopping policy chosen:**
  Bandit Policy is also less resource intensive and time consuming. If a run's performance is outside the best run's slack_factor, the run is early terminated -- saving time and resources.





### Table of Contents

1. [Summary](#summary)
2. [Scikit-learn Pipeline](#pipeline)
3. [AutoML](#automl)
4. [Pipeline Comparison](#comparison)
5. [Future work](#future)
6. [Clean up](#clean)
7. [Useful Resources](#clean)



## Summary<a name="summary"></a>
This project is based on a bank's marketing campaign and the goal of this project is to predict if a customer will sign up for a new term deposit offering.  The dataset used in this project contains data about a bank's marketing campaign including bank client's demographic information.  

Azure AutoML produced the best performing model which was VotingEnsemble with an accurary .91756

## Scikit-learn Pipeline<a name="pipeline"></a>

#### Pipeline Architecture
* Prepare Data
* Download the dataset [Data](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) and convert into   TabularDatasetFactory dataset.
* Clean the dataset (clean_data located in train.py)
* Split data into training and test sets (80/20)
* Utilize a Scikit-learn Logistic Regression Model for Classification
   
#### Hyperparamater Tuning
hyperdrive_config = HyperDriveConfig (
hyperparameter_sampling=RandomParameterSampling(
{"--C":choice(0.5,1.0), --max_iter":choice(50,100,150)})  ,
primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
primary_metric_name='Accuracy',
policy=BanditPolicy(evaluation_interval=1, slack_factor=0.1),
max_total_runs=8, 
max_concurrent_runs=4,
estimator=SKLearn(source_directory='.', entry_script='train.py', compute_target=cluster_name))
           
#### Classifcation Algorithim
* Logistic Regression
   

**Benefits of the parameter sampler chosen:**
  RandomParameterSampling supports continous and discrete hyperparamters.  It is also less resource intensive and time consuming.

**Benefits of the early stopping policy chosen:**
  Bandit Policy is also less resource intensive and time consuming. If a run's performance is outside the best run's slack_factor, the run is early terminated -- saving time and resources.

## AutoML<a name="automl"></a>
* Download the dataset [Data](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) and convert into   TabularDatasetFactory dataset.
* Clean the dataset (clean_data located in train.py)
* Split data into training and test sets (80/20)
* Configure AutoML
* Save Best Model

#### AutoML Config
AutoMLConfig(
    experiment_timeout_minutes=30,
    task= 'classification',
    primary_metric='accuracy',
    training_data= train_data,
    label_column_name= 'y',
    n_cross_validations= 4, compute_target = compute_target)
    
    
#### Best Model 
* VotingEnsemble with an accurary of 91.756%

## Pipeline comparison<a name="comparison"></a>
AutoML had the best accuary with VotingEnsemble @ 91.756% and Hyperdrive received an accuracy score of 90.94%.  AutoML was able to find the best alogrithm and hyper parameter settings to achieve the higher accuracy score.

![Pipeline Comparison](pipeline.PNG)

#### Best Model Summary:
![Pipeline Comparison](best_model_summary.PNG)

#### Best Model Statistics:
![Pipeline Comparison](best_model.PNG)


## Future Work<a name="future"></a>
* Enable Onnx Compatible Models
* Replace Deprecated SkLearn Estimator

## Proof of cluster clean up<a name="clean"></a>
![Proof of Cluster Clean up](delete_compute_target.PNG)

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

# Heart Failure Prediction
## Udacity Azure Machine Learning Nanodegree Capstone Project

Heart failure is a common event caused by Cardiovascular diseases which is the #1 cause of death globally, taking an estimated 17.9 million lives each year (31% of all deaths worlwide).  This project uses [Heart Failure Clinical Data](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data) to predict a death event caused by heart failure.  This dataset contains 12 clinical features for predicting death events based on medical records of 299 patients.

In this Udacity Azure Machine Learning Nanodegree Capstone project, I created two models: one using Automated ML (denoted as AutoML from now on) and one customized model whose hyperparameters are tuned using HyperDrive. I then compared the performance of both the models and deployed the best performing model.  This project used an [external datasets](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data) and trained a model using different tools available in the AzureML framework as well as deploying the model as a web service.

## Project Workflow
![project_workflow](./images/workflow.PNG)


### Project Main Steps:

1. [Project Set Up and Installation](#setup)
2. [Dataset](#dataset)
	* Overview
	* Task
	* Access
3. [Automated ML](#automl)
	* Overview
	* Results
4. [Hyperparameter Tuning](#hyper)
	* Overview
	* Results
5. [Model Deployement](#deploy)
	* Overview
	* Results
7. [Create and publish a pipeline](#pipeline)
8. [Documentation Video](#video)
9. [Future Improvements](#future)


## Step 1: Project Set Up and Installation<a name="setup"></a>

* Create a new workspace in Microsoft Azure Machine Learning Studio
* Create Compute Instance called automl-inst using STANDARD_DS12_V2
* Use the Compute Instance terminal and type:
```
git clone https://github.com/camille-wilkens/AZ_Capstone.git --depth 1
```
* Open automl.ipynb and execute all the cells
* Open hyperparameter_tuning.ipynb and execute all the cells


## Step 2: Dataset<a name="dataset"></a>

### Overview
This project uses an external dataset from Kaggle - [Heart Failure Clinical Data](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data). This dataset contains 12 features that can be used to predict mortality by heart failure. It contains medical records of 299 patients 

```
Dataset from Davide Chicco, Giuseppe Jurman: “Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making 20, 16 (2020)
```

### Features & Target

| Name | Description | Type|
|:--------------:|:-------------:|:--------------:|
| age | Age of Patient| Years|
| anaemia | Decrease of red blood cells or hemoglobin| (boolean)|
| creatinine_phosphokinase | Level of the CPK enzyme in the blood| (mcg/L)|
| diabetes | If the patient has diabetes| (boolean)|
| ejection_fraction| Percentage of blood leaving the heart at each contraction| (percentage)|
| high_blood_pressure| If the patient has hypertension| (boolean)|
| platelets| Platelets in the blood | (kiloplatelets/mL)|
| serum_creatinine| Level of serum creatinine in the blood  | (mg/dL)|
| serum_sodium| Level of serum sodium in the blood   | (mEq/L)|
| sex| Woman or man  | binary|
| smoking| If the patient smokes or not  | (boolean)|
| time| Follow-up period  | (days)|
| TARGET: DEATH_EVENT| If the patient deceased during the follow-up period | (boolean)|

#### Task

* The task involves predicting the Death Event (if the patient deceased during the follow-up period) based on the other 12 features available (see above).  Using Classification, AutoML will predict Death Event and the Hyperdrive model will use Logistic Regression to predict Death Event

#### Access

In AutoML, a Jupyter Notebook (automl.ipynb) reads the dataset using Dataset.Tabular.from_delimited_files and registers the data in the workspace.
```
found = False
key = "Heart_Prediction_Dataset"
description_text = "Heart Prediction Dataset"

if key in ws.datasets.keys(): 
        found = True
        dataset = ws.datasets[key] 

if not found:
        # Create AML Dataset and register it into Workspace
        example_data = 'https://raw.githubusercontent.com/camille-wilkens/AZ_Capstone/main/heart_failure_clinical_records_dataset.csv'
        dataset = Dataset.Tabular.from_delimited_files(example_data)        
        #Register Dataset in Workspace
        dataset = dataset.register(workspace=ws,
                                   name=key,
                                   description=description_text)
```
                        
                        
In the Hyperdrive model, a Jupyter Notebook (hyperparameter_tuning.ipynb) reads the dataset using train.py which creates a TabularDataset using TabularDatasetFactory. 

## Step 4: Automated ML<a name="automl"></a>
#### Overview 
`automl` settings and configuration utlized in this experiment  
```
automl_settings = {
    "experiment_timeout_minutes": 30,
    "max_concurrent_iterations": 5,
     "n_cross_validations": 25,
     "primary_metric" : 'AUC_weighted'
}
```

```
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="Survived",   
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
```

#### Results
VotingEnsemble with an accurary of 91.756% was tehe best model


*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

#### Screenshots
Remeber to provide screenshots of the `RunDetails` widget best model trained with it's parameters.

#### RunDetails widget (shows progress of training runs of the different experiements): 
![RunDetails Widget](rundetails.PNG)


#### Best Model with Run ID (VotingEnsemble): 
![Best Model Summary](best_model_summary.PNG)

#### Improvement Areas

AutoML alerted to Class balancing detection and changing the accuary metric to another type could address the imbalanced data. Adding data to ensure each class has a good representation in the dataset, measured by the number and ratio of samples could address this issue as well.


## Step 4: Hyperparameter Tuning<a name="hyper"></a>
#### Overview
I utilized a Scikit-learn Logistic Regression Model for Classification on this HyperDrive experiment.


* Hyperparameter Sampling
I used RandomParameterSampling as it supports continous and discrete hyperparamters.  Another key benefit of using RandomParameterSampling, is its less resource intensive and time consuming and supports early termination.

"--C":choice(0.5,1.0),     
"--max_iter":choice(50,100,150)

```
* Early Stopping Policy
* The Early Stopping policy, I utilized was the Bandit Policy is also less resource intensive and time consuming.  If a run's performance is outside the best run's slack_factor, the run is early terminated -- saving time and resources.


#### Model Used & Why 

* Utilized a Scikit-learn Logistic Regression Model for Classification
   
#### Hyperparamater Tuning
```
hyperdrive_config = HyperDriveConfig (
hyperparameter_sampling=RandomParameterSampling(
{"--C":choice(0.5,1.0), --max_iter":choice(50,100,150)})  ,
primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
primary_metric_name='Accuracy',
policy=BanditPolicy(evaluation_interval=1, slack_factor=0.1),
max_total_runs=8, 
max_concurrent_runs=4,
estimator=SKLearn(source_directory='.', entry_script='train.py', compute_target=cluster_name))
```
           
#### Results

Hyperdrive experiment received an accuracy score of 90.94%. Add best run metrics : {'Regularization Strength:': 0.1, 'Max iterations:': 150, 'Accuracy': 0.6059113300492611}

The parameters used were ---

#### Screenshots
Remeber to provide screenshots of the `RunDetails` widget best model trained with it's parameters.

#### RunDetails widget (shows progress of training runs of the different experiements): 
![RunDetails Widget](rundetails.PNG)


#### Best Model with Run ID (VotingEnsemble): 
![Best Model Summary](best_model_summary.PNG)

#### Improvement Areas
Changing the sampling type to either GRID or Bayesian sampling could improve the model as well as increasing the maximum run size.


## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
#### Overview


#### [Overview for deploying a model](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python)

* Register the model.
* Prepare an inference configuration.
* Prepare an deployment configuration.
* Deploy the model as a web service to the cloud.
* Test the resulting web service.

#### Instructions on how to query the endpoint



## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

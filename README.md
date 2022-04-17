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
5. [Model Deployment](#deploy)
	* Overview
	* Instructions on how to query the endpoint
6. [Documentation Video](#video)
7. [Future Improvements](#future)
8. [Acknowledgements](#ack)

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

## Step 3: Automated ML<a name="automl"></a>
### Overview 
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
                             label_column_name="DEATH_EVENT",   
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
```

### Results
VotingEnsemble with an accurary of 85.8% and AUC Weighted of 92.3% was the best model

### Best Run
```
best_run,fitted_model = remote_run.get_output()
best_run_metrics = best_run.get_metrics() 

print("Best Run:",best_run.id)
print(best_run)
print("Fitted Model:", fitted_model)
print("Fitted Model Steps:")
print(fitted_model.steps)
print("Best Run Metrics")
print(best_run_metrics)

for i in best_run_metrics:
    x = best_run_metrics[i]
    print(i,x )
```

```
Best Run: AutoML_11620fab-74d8-4a45-b233-a596d2fcec2f_38
Run(Experiment: ml-experiment-1,
Id: AutoML_11620fab-74d8-4a45-b233-a596d2fcec2f_38,
Type: azureml.scriptrun,
Status: Completed)
Fitted Model: Pipeline(memory=None,
         steps=[('datatransformer',
                 DataTransformer(enable_dnn=False, enable_feature_sweeping=True, feature_sweeping_config={}, feature_sweeping_timeout=86400, featurization_config=None, force_text_dnn=False, is_cross_validation=True, is_onnx_compatible=False, observer=None, task='classification', working_dir='/mnt/batch/tasks/shared/LS_root/mount...
                 PreFittedSoftVotingClassifier(classification_labels=array([0, 1]), estimators=[('1', Pipeline(memory=None, steps=[('maxabsscaler', MaxAbsScaler(copy=True)), ('xgboostclassifier', XGBoostClassifier(n_jobs=1, problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), random_state=0, tree_method='auto'))], verbose=False)), ('0', Pipeline(memory=None, steps=[('maxabsscaler', MaxAbsScaler(copy=True)), ('lightgbmclassifier', LightGBMClassifier(min_data_in_leaf=20, n_jobs=1, problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), random_state=None))], verbose=False)), ('4', Pipeline(memory=None, steps=[('standardscalerwrapper', StandardScalerWrapper(copy=True, with_mean=False, with_std=True)), ('lightgbmclassifier', LightGBMClassifier(boosting_type='gbdt', colsample_bytree=0.4955555555555555, learning_rate=0.09473736842105263, max_bin=140, max_depth=6, min_child_weight=0, min_data_in_leaf=0.08276034482758622, min_split_gain=0.10526315789473684, n_estimators=25, n_jobs=1, num_leaves=164, problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), random_state=None, reg_alpha=0.3157894736842105, reg_lambda=0.3157894736842105, subsample=0.5942105263157895))], verbose=False)), ('20', Pipeline(memory=None, steps=[('standardscalerwrapper', StandardScalerWrapper(copy=True, with_mean=False, with_std=False)), ('xgboostclassifier', XGBoostClassifier(booster='gbtree', colsample_bytree=0.9, eta=0.1, gamma=0, max_depth=6, max_leaves=3, n_estimators=25, n_jobs=1, objective='reg:logistic', problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), random_state=0, reg_alpha=0, reg_lambda=0.7291666666666667, subsample=0.5, tree_method='auto'))], verbose=False)), ('23', Pipeline(memory=None, steps=[('maxabsscaler', MaxAbsScaler(copy=True)), ('lightgbmclassifier', LightGBMClassifier(boosting_type='goss', colsample_bytree=0.5944444444444444, learning_rate=0.026323157894736843, max_bin=310, max_depth=-1, min_child_weight=3, min_data_in_leaf=1e-05, min_split_gain=0.7894736842105263, n_estimators=50, n_jobs=1, num_leaves=131, problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), random_state=None, reg_alpha=0.3684210526315789, reg_lambda=1, subsample=1))], verbose=False)), ('29', Pipeline(memory=None, steps=[('maxabsscaler', MaxAbsScaler(copy=True)), ('lightgbmclassifier', LightGBMClassifier(boosting_type='gbdt', colsample_bytree=0.2977777777777778, learning_rate=0.06842421052631578, max_bin=130, max_depth=8, min_child_weight=1, min_data_in_leaf=0.09310413793103449, min_split_gain=0.631578947368421, n_estimators=100, n_jobs=1, num_leaves=119, problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), random_state=None, reg_alpha=1, reg_lambda=0.47368421052631576, subsample=0.8415789473684211))], verbose=False)), ('33', Pipeline(memory=None, steps=[('sparsenormalizer', Normalizer(copy=True, norm='l2')), ('xgboostclassifier', XGBoostClassifier(booster='gbtree', colsample_bytree=1, eta=0.01, gamma=0, grow_policy='lossguide', max_bin=63, max_depth=10, max_leaves=511, n_estimators=100, n_jobs=1, objective='reg:logistic', problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), random_state=0, reg_alpha=1.5625, reg_lambda=0.625, subsample=0.8, tree_method='hist'))], verbose=False))], flatten_transform=None, weights=[0.15384615384615385, 0.38461538461538464, 0.15384615384615385, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693]))],
         verbose=False)
```


#### Screenshots

#### RunDetails widget (shows progress of training runs of the different experiments): 
![RunDetails Widget](./images/automl_best_run_details.PNG)

#### Best Model with Run ID (VotingEnsemble): 
![Best Model Summary](./images/automl_best_model2.PNG)
![Best Model Summary](./images/automl_best3.PNG)
![Best Model Summary](./images/automl_best4.PNG)



#### Improvement Areas

* Increase the number of cross validations
* Spend more time on preparing the data, adding more patient data
* Try a different primary metric
* Increase the Experiment Timeout


## Step 4: Hyperparameter Tuning<a name="hyper"></a>
#### Overview
I utilized a Scikit-learn Logistic Regression Model for Classification on this HyperDrive experiment.


* Hyperparameter Sampling
I used RandomParameterSampling as it supports continous and discrete hyperparamters.  Another key benefit of using RandomParameterSampling, is its less resource intensive and time consuming and supports early termination.

```

"--C":choice(0.5,1.0),     
"--max_iter":choice(50,100,150)

```
* Early Stopping Policy
* The Early Stopping policy, I utilized was the Bandit Policy is also less resource intensive and time consuming.  If a run's performance is outside the best run's slack_factor, the run is early terminated -- saving time and resources.


#### Hyperparamater Tuning
```
hyperdrive_config = HyperDriveConfig (
        hyperparameter_sampling=ps,
        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
        primary_metric_name='Accuracy',
        policy=policy,
        max_total_runs=8, 
        max_concurrent_runs=4,
        estimator=est)
```
           
#### Results

Hyperdrive experiment received an accuracy score of 80%
```
best_run.get_file_names()
best_run_metrics

```
```
{'Regularization Strength:': 0.5, 'Max iterations:': 150, 'Accuracy': 0.8}
```
#### Screenshots

#### RunDetails widget (shows progress of training runs of the different experiements): 
![RunDetails Widget](./images/hyper_rundetails2.PNG)
![RunDetails Widget](./images/hyper_rundetails3.PNG)

#### Best Run
![RunDetails Widget](./images/hyper_best.PNG)
![RunDetails Widget](./images/hyper_best2.PNG)
![RunDetails Widget](./images/hyper_best3.PNG)


#### Improvement Areas
Changing the sampling type to either GRID or Bayesian sampling could improve the model.


## Step 5: Model Deployment<a name="deploy"></a>

#### [Overview for deploying a model](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python)

* Register the model.
* Prepare an inference configuration.
* Prepare an deployment configuration.
* Deploy the model as a web service to the cloud.
* Test the resulting web service.

#### Instructions on how to query the endpoint

The following cell shows how to query the endpoint:
```

import json
import requests


test = {
  "data": [ {
      "age": 90,
      "anaemia": 0,
      "creatinine_phosphokinase": 500, 
      "diabetes":  1,
      "ejection_fraction": 150,
      "high_blood_pressure": 0,
      "platelets": 100000,
      "serum_creatinine": 2.75,
      "serum_sodium": 140,
      "sex": 1,
      "smoking": 0,
      "time": 500 },
      
      {"age": 45,
      "anaemia": 1,
      "creatinine_phosphokinase": 500, 
      "diabetes":  1,
      "ejection_fraction": 50,
      "high_blood_pressure": 1,
      "platelets": 100000,
      "serum_creatinine": 1.75,
      "serum_sodium": 140,
      "sex": 0,
      "smoking": 1,
      "time": 213}
  ]
}


test_data = json.dumps(test)
```

Returns the following:
```
[0, 0]
```
#### Screenshots
![Best Model Summary](./images/endpoint.PNG)
![Best Model Summary](./images/deployed_web_service.PNG)
![Best Model Summary](./images/test_web_service.PNG)
![Best Model Summary](./images/test_endpoint.PNG)




## Step 7: Documentation Video<a name="video"></a>

[YouTube Video](https://www.youtube.com/watch?v=sb2C0TIpM04)


## Acknowledgements<a name="ack"></a>


#### Citation
Davide Chicco, Giuseppe Jurman: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making 20, 16 (2020). (link)

#### License
CC BY 4.0


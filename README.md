# P4ML-UI
P4ML User Interaction project.


## Description
Human-guided machine learning (HGML) is a hybrid approach where a user interacts with an AutoML system and tasks it to explore different problem settings that reflect the user’s knowledge about the data available. 


## How to run the planner
go to the direcotry /P4ML-UI/dsbox-ta2/python
```
python ta2-search config.json restriction.json
```

## Restriction File
>/P4ML-UI/dsbox-ta2/python/Planner requirements in restriction file.docx

This document explains the schema in restriction file of human-guided machine learning (HGML), in which the user provide their restriction requirement to our system.

### Restriction Schema
-	include_model [status: implemented]
     - Definition: this flag allows you to customize the model being included in the solution pipelines.
     - Parameters: Array with the ids of the primitive models to be included. All primitives have to be registered as part of the primitive library of the TA2 system. 
     - Default value: if empty, the TA2 system will explore all possible pipelines.
     - If include_model and exclude_model has the same models, the TA2 system will explore all possible pipelines.
     - Example: "include_model":["LinearSVC","LogisticRegression"]

-	exclude_model [status: implemented]
     - Definition: this flag allows you to customize the model being excluded in the solution pipelines.
     - Parameters: Array with the ids of the primitive models to be excluded. All primitives have to be registered as part of the primitive library of the TA2 system. 
     - Default value: if empty, the TA2 system will explore all possible pipelines.
     - If include_model and exclude_model has the same models, the TA2 system will explore all possible pipelines.
     - Example: "exclude_model":["LinearSVC","LogisticRegression"]

-	include_feature_generation [status: implemented]
     - Definition: this flag allows you to provide a rule that generate the feature
     - Parameters: Array with the names of feature generation primitive. All primitives have to be registered as part of the primitive library of the TA2 system.
     - Default value: if empty, the TA2 system will explore all possible pipelines.
     - Example: "include_feature_generation":[""]

-	use_imputation_method [status: implemented]
     - Definition: this flag allows you to use specific imputation method for missing values
     - Parameters: imputation method name. All primitives have to be registered as part of the primitive library of the TA2 system.
     - Default value: if empty, the TA2 system will use “mean” to impute missing values.
     - Example: "use_imputation_method":"median"/"most frequent"
     
 -	replace_model [status: ongoing]
     - Definition: this flag allows you to have two or more solutions with different models but same other steps.
     - Parameters: Array contains the model you want to replace. 
     - Default value: if empty, the TA2 system will explore all possible pipelines
     - Example: "replace_model": {“replace_model”: [“LogisticRegression”],”new_model”:[“RandomForestClassifier”]}

-	include_variables [status: not implemented]
     - Definition: this flag allows you to customize the variables (columns) being included in the solution.
     - Parameters: Array with the ids of the variables to be included. 
     - Default value: if empty, the TA2 system will use all variables.
     - Example: "include_variables":[]

-	exclude_variables [status: not implemented]
     - Definition: this flag allows you to customize the variables being excluded in the solution.
     - Parameters: Array with the ids of the variables to be included. 
     - Default value: if empty, the TA2 system will use all variables.
     - Example: "exclude_variables":[]

-	include_instances [status: not implemented]
     - Definition: this flag allows you to customize the instances (rows) being included in the solution.
     - Parameters: Array with the ids of the instances to be included. 
     - Default value: if empty, the TA2 system will use all instances.
     - Example: "include_instances":[]

-	exclude_instances [status: not implemented]
     - Definition: this flag allows you to customize the instances being included in the solution.
     - Parameters: Array with the ids of the instances to be excluded. 
     - Default value: if empty, the TA2 system will use all instances.
     - Example: "exclude_instances":[]

-	define_variable_weight [status: not implemented]
     - Definition: this flag allows you to specify the priority/ relative weight of variables.
     - Parameters: dictionary includes the ids of the variables and their priority . 
     - Default value: if empty, the TA2 system will set each variable the same priority.
     - Example: "define_variable_weight":{“variable_id“: ,”priority”:1}

-	select_training_and_test_data [status: not implemented]
     - Definition: this flag allows you to select training and testing data, optionally with cross-validation specifications
     - Parameters: dictionary includes the ids of instances to be the training and testing data, with optional cross-validation method 
     - Default value: if empty, the TA2 system will use the default method to get training and testing data.
     - Example: "select_training_and_testing_data":{“training_data“:(1,1000) ,”testing_data”:(1001,1200), “cross_validation”:”k fold”}

-	use_specific_parameter_for_model [status: not implemented]
     - Definition: this flag allows you to specify a model and the parameter values desired
     - Parameters: An array of dictionaries, including the model and its parameters, the parameters are stored in a dictionary
     - Default value: if empty, the TA2 system will use the default parameters
     - Example: "use_specific_parameter_for_model": [{"model":"DecisionTreeClassifier","parameter":{},”parameter_value:}]

-	include_class_of_model [status: not implemented]
     - Definition: this flag allows you to specify the class of model desired
     - Parameters: An array of class of model. All primitives have to be registered as part of the primitive library of the TA2 system.
     - Default value: if empty, the TA2 system will explore all possible pipelines
     - Example: "include_class_of_model": []

-	include_class_of_data_preparation_method [status: not implemented]
     - Definition: this flag allows you to specify the class of data preparation method
     - Parameters: An array of class of data preparation method. All primitives have to be registered as part of the primitive library of the TA2 system.
     - Default value: if empty, the TA2 system will explore all possible pipelines
     - Example: "include_class_of_data_preparation_method": []

-	include_statistical_test [status: not implemented]
     - Definition: this flag allows you to request a particular statistic test and parameters
     - Parameters: An array of statistical test. All primitives have to be registered as part of the primitive library of the TA2 system.
     - Default value: if empty, the TA2 system will explore the default statistical test
     - Example: "include_statistical_test": []

-	results_after_step [status: not implemented]
     - Definition: this flag allows you to request results after any step in a solution
     - Parameters: step’s name. All primitives have to be registered as part of the primitive library of the TA2 system.
     - Default value: if empty, the TA2 system will show the final result
     - Example: "results_after_step": “DataPreprocessing”

-	compare_solutions [status: not implemented]
     - Definition: this flag allows you to generate comparative explanations for two given solutions
     - Parameters: dictionary includes model name and different solutions. All primitives have to be registered as part of the primitive library of the TA2 system.
     - Default value: if empty, the TA2 system will show the final result
     - Example: "compare_solutions": {"model":,"data_preparation_method":[]}

-	compare_models [status: not implemented]
     - Definition: this flag allows you to generate comparative models.
     - Parameters: array includes model names. All primitives have to be registered as part of the primitive library of the TA2 system.
     - Default value: if empty, the TA2 system will only show the final result
     - Example: "compare_models": []


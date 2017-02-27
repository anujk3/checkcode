# Home Work - 2
## Anuj Katiyal (ak3979), Vinayak Bakshi (vb2424)
---

# Task - 1: Predict the Monthly rent of an Apartment

## Step 1: Data Cleaning & Pre-processing

* We first identified variables relevant to our model by looking through the data description files. The variable *uf17 (Monthly Contract Rent)* is our predictor variable. We excluded all the recoded variables. Only features that *apply to pricing an apartment that is not currently rented* were manually selected and placed in a pickle file. Given below:

*      'boro', 'uf1_1', 'uf1_2', 'uf1_3', 'uf1_4', 'uf1_5', 'uf1_6', 'uf1_7',
       'uf1_8', 'uf1_9', 'uf1_10', 'uf1_11', 'uf1_12', 'uf1_13', 'uf1_14',
       'uf1_15', 'uf1_16', 'uf1_35', 'uf1_17', 'uf1_18', 'uf1_19', 'uf1_20',
       'uf1_21', 'uf1_22', 'sc23', 'sc24', 'sc36', 'sc37', 'sc38', 'sc114',
       'sc120', 'sc121', 'uf5', 'uf6', 'sc127', 'uf7', 'sc134', 'uf7a', 'uf9',
       'sc141', 'uf8', 'sc143', 'sc144', 'uf10', 'uf48', 'sc147', 'uf11',
       'sc149', 'sc173', 'sc171', 'sc150', 'sc151', 'sc152', 'sc153', 'sc154',
       'sc155', 'sc156', 'sc157', 'sc158', 'sc159', 'uf12', 'sc161', 'uf13',
       'uf14', 'sc164', 'uf15', 'sc166', 'uf16', 'sc174', 'uf64', 'uf17',
       'sc181', 'sc541', 'sc184', 'sc542', 'sc543', 'sc544', 'uf17a', 'sc185',
       'sc186', 'sc197', 'sc198', 'sc187', 'sc188', 'sc571', 'sc189', 'sc190',
       'sc191', 'sc192', 'sc193', 'sc194', 'sc196', 'sc548', 'sc549', 'sc550',
       'sc199', 'sc575', 'new_csr', 'cd'

* The variables are mapped into Quantitative and Categorical types.

* All samples that correspond to 'not reported' values for uf17 (Monthly Contract Rent) are deleted

* We exclude Quantitative variables with percent 'not reported'/not applicable > 90 or ones which are direct recodes of our predictor variable uf17. Variables like ['uf5', 'uf6', 'uf7', 'uf7a', 'uf9', 'uf8', 'uf10', 'uf13', 'uf14', 'uf15', 'uf16', 'uf17a'] are removed.

* Quantitative variables uf12 (Monthly Cost) and uf64 (Yearly Assistance amount) are included in the model. The values 'not reported' were replaced with NaN for subsequent imputation.

* 'Not Reported' and 'Not Applicable' values of categorical variables are imputed appropriately based on the use case.

* Dummy variables are created for all the categorical variables.

### Scaling Data:
* The data is scaled using MaxAbsScaler. Alternate scalers like Robust Scaler, Normalizer were tried to improve the model. Since most features were categorical, scaling did not make any difference to model performance.

### Imputing Data:
* The continuous variables are imputed using Median values. We also tried FancyImpute - MICE for imputation but since we were not able to include it in the pipeline we used a simple Median imputation for all continuous variables.


## Step 2: Model Creation

* We fit LASSO, Ridge and Elastic Net regression models using a 5-fold cross validation.

* Grid Search is used to obtain best hyper-parameters for each model. Alpha for LASSO is obtained as 1.25

* A pipeline is created with a scaler, Imputer and Model_fit within the cross-validation function to ensure there is no information leak.

### Model Output

* We obtain the best *R-square of  52.78% for LASSO model*  using cross-validation.
* Some of the best predictors for our model are *Borough type, sub-borough type, Condition of Building, Number of units in the building, Number of Stories in Building*


## Step 3: Model Validation

The model output is tested on 20% of the data that was kept for testing. We obtain an *R-square score of 52.6%* on the test set.

We also conduct residual analysis (residual Vs Fitted) and observe the residuals to be randomly scattered around the x-axis. Although we observe some outliers which need to be taken care of

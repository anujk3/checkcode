import pandas as pd
import scipy
import pickle
import numpy as np
import operator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import KFold


def readAndCleanDf():
    df = pd.read_csv("https://ndownloader.figshare.com/files/7586326")

    ## Reading the saved columns from analysis in a pickle
    pk1_file = open('df_column_iter1.p', 'rb')
    df_column_iter1 = pickle.load(pk1_file)
    pk1_file.close()

    ## Removing the recodes
    start_recode_index = df_column_iter1.index("uf19")
    df_column_iter1 = df_column_iter1[:start_recode_index]

    ## Adding variables that help in improving the model further
    df = df[df_column_iter1 + ["new_csr", "cd"]]

    # Remove the uf17 rows where the value is 99999
    df = df[df["uf17"] != 99999]

    ## Ignore Quantitative Variables that have been included because of a lot of NaNs
    rm_quant_vars = ['uf5', 'uf6', 'uf7', 'uf7a', 'uf9', 'uf8', 'uf10', 'uf13', 'uf14', 'uf15', 'uf16', 'uf17a']
    df.drop((rm_quant_vars), axis=1, inplace=True)

    Y = df["uf17"]
    df.drop(["uf17"], axis=1, inplace=True)

    ## Quantitative variables to consider
    quant_vars = ['uf12', 'uf64']

    return df, quant_vars, Y


def score_rent():
    """
    Returns the R^2 value
    """

    df, quant_vars, Y = readAndCleanDf()

    ## Quantitative Variables Modified before Model, keep the uf17 intact before making dummies
    df["uf64"].replace(9999, 0, inplace=True)
    df["uf64"].replace(9998, np.nan, inplace=True)
    df["uf12"].replace(9999, 0, inplace=True)

    encoded_df = pd.DataFrame()
    for i, column in enumerate(df):
        if column not in quant_vars:
            out = pd.get_dummies(df[column], prefix=column)
            if i == 0:
                encoded_df = out
            else:
                encoded_df = pd.concat([encoded_df, out], axis=1)
        else:
            encoded_df = pd.concat([encoded_df, df[column]], axis=1)

    # print(encoded_df.shape)

    X = encoded_df.values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=7)

    # Standardize the dataset
    pipelines = []
    pipelines.append(("ScaledLASSO", Pipeline([("Imputer", Imputer(strategy="median")),
                                               ("Scaler", MaxAbsScaler()),
                                               ("LASSO", Lasso())])))
    pipelines.append(("ScaledEN", Pipeline([("Imputer", Imputer(strategy="median")),
                                            ("Scaler", MaxAbsScaler()),
                                            ("EN", ElasticNet())])))
    pipelines.append(("ScaledRIDGE", Pipeline([("Imputer",
                                                Imputer(strategy="median")),
                                                ("Scaler", MaxAbsScaler()),
                                                ("RIDGE", Ridge())])))
    results = []
    names = []
    num_folds = 5
    seed = 7
    select_model = {}
    for name, model in pipelines:
        kfold = KFold(n_splits=5, random_state=7)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
        results.append(cv_results)
        names.append(name)
        select_model[name] = cv_results.mean()
        #msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        #print(msg)

    take_best_train_model = sorted(select_model.items(), key=operator.itemgetter(1), reverse=True)

    if "LASSO" in take_best_train_model[0][0]:
        param_grid = {'lasso__alpha': np.logspace(.1, 10, 10)}
        pipe = make_pipeline(Imputer(strategy="median"), MaxAbsScaler(), Lasso())

    elif "RIDGE" in take_best_train_model[0][0]:
        param_grid = {'ridge__alpha': np.logspace(.1, 10, 10)}
        pipe = make_pipeline(Imputer(strategy="median"), MaxAbsScaler(), Ridge())
    else:
        param_grid = {'elasticnet__alpha': np.logspace(.1, 10, 10)}
        pipe = make_pipeline(Imputer(strategy="median"), MaxAbsScaler(), ElasticNet())

    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=kfold)
    grid_result = grid.fit(X_train, y_train)
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    y_pred = grid.predict(X_test)
    # print(y_pred)
    return r2_score(y_test, y_pred), X_test, y_test, grid

    # print('mean abs error', mean_absolute_error(y_test, y_pred))
    # print('mean squared error', mean_squared_error(y_test, y_pred))
    # print('median absolute error', median_absolute_error(y_test, y_pred))
    # print(grid.score(X_test, y_test))


def predict_rent(X_test, y_test, best_grid):
    """
    Returns the test data, the true labels and your predicted labels (all as numpy arrays)
    """
    y_pred = best_grid.predict(X_test)
    return X_test, y_test, y_pred


r2_value, X_test, y_test, grid_value = score_rent()
print(r2_value)

test_data, true_labels, predicted_labels = predict_rent(X_test, y_test, grid_value)

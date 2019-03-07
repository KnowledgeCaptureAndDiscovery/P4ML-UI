import pandas as pd
import numpy as np

def popular_value(array):
    """
    array: 1D array
    """
    cate = dict()
    for cell in array:
        if (cell in cate):
            cate[cell] += 1
        else:
            cate[cell] = 1

    max_freq = 0
    popular = None
    for key in cate.keys():
        if (cate[key] > max_freq):
            popular = key

    return popular

def myImputer(data, value="zero", verbose=0):
    """
    INPUT:
    data: numpy array, matrix
    value:    string: "mean", "min", "max", "zero", "gaussian"
    """
    index = pd.isnull(data)
    
    data_imputed = np.copy(data)
    data_drop = data[np.logical_not(index)]   #drop nan from data
    inputed_value = 0
    if (value == "zero"):
        inputed_value = 0
    elif (value == "mean"):
        inputed_value = np.mean(data_drop)
    elif (value == "max"):
        inputed_value = np.max(data_drop)
    elif (value == "min"):
        inputed_value = np.min(data_drop)
    elif (value == "new"):
        inputed_value = 0   # 0 is the value that never happens in our categorical map
    elif (value == "popular"):
        inputed_value = popular_value(data_drop)
    # special type of imputed, just return after imputation
    elif (value == "knn"):
        from fancyimpute import KNN
        data_clean = KNN(k=5).complete(data)
        return data_clean
    else:
        raise ValueError("no such impute strategy: {}".format(value))

    data_imputed[index] = inputed_value

    if verbose: print("imputed missing value: {}".format(inputed_value))
    return data_imputed


def imputeData(data, missing_col_id, imputation_strategies,verbose=0):
    """
    impute the data using permutations array.
    INPUT:
    data: numpy array, matrix
    value:    string: "mean", "min", "max", "zero", "gaussian"
    """
    data_clean = np.copy(data)

    for i in range(len(imputation_strategies)):
        strategy = imputation_strategies[i]
        col_id = missing_col_id[i]

        data_clean[:,col_id] = myImputer(data[:,col_id], strategy)


    return data_clean

def bayeImpute(data, target_col, verbose=0):
    '''
    currently, BayesianRidge.
    return the imputated data, and model coefficient
    '''

    from sklearn.linear_model import BayesianRidge, LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    model = BayesianRidge()
    # model = LinearRegression()
    # model = RandomForestRegressor()

    original_data = np.copy(data)

    target = data[:, target_col]
    data = np.delete(data, obj=target_col, axis=1)  #remove the missing-value column
    mv_mask = pd.isnull(target)
    if verbose: print("number of imputated cells: {}".format(sum(pd.isnull(original_data[:,target_col]))))

    x_test = data[mv_mask]
    x_train = data[~mv_mask]
    y_train = target[~mv_mask]
    # check if valid to regression: wether only one value exist in target. 
    # If happen, use default "mean" method (which is all same)
    is_other_value = False in (y_train == y_train[0])
    if (not is_other_value):
        model = "mean"
        original_data[mv_mask, target_col] = y_train[0] * len(mv_mask)
        return original_data, model 

    model.fit(x_train, y_train)
    result = model.predict(x_test)
    original_data[mv_mask, target_col] = result #put the imputation result back to original data, following the index

    # print("coefficient: {}".format(model.coef_))
    return original_data, model



def transform(data, target_col, model, verbose=0):
    '''
    currently, BayesianRidge.
    return the imputated data, and model coefficient
    '''

    original_data = np.copy(data)

    target = data[:, target_col]
    data = np.delete(data, obj=target_col, axis=1)  #remove the missing-value column
    mv_mask = pd.isnull(target)
    if verbose: print("number of imputated cells: {}".format(sum(pd.isnull(original_data[:,target_col]))))

    x_test = data[mv_mask]
    x_train = data[~mv_mask]
    y_train = target[~mv_mask]

    result = model.predict(x_test)
    original_data[mv_mask, target_col] = result #put the imputation result back to original data, following the index

    # print("coefficient: {}".format(model.coef_))
    return original_data

def df2np(data, missing_col_id=[], verbose=0):
    """
    helper function: convert dataframe to np array;
        in the meanwhile, provide the id for missing column
    """
    counter = 0

    # 1. get the id for missing value column
    missing_col_name = []
    for col_name in data:
        num = sum(pd.isnull(data[col_name]))
        if (num > 0):
            missing_col_id.append(counter)
            missing_col_name.append(col_name)
        counter += 1

    if (verbose>0): print("missing column name: {}".format(missing_col_name))

    data = data.values  #convert to np array

    return data

import numpy as np
import sklearn
from pandas import DataFrame
from typing import List
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y, check_is_fitted


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, "weights_")

        # TODO: Calculate the model prediction, y_pred

        y_pred = np.matmul(X,self.weights_)

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO:
        #  Calculate the optimal weights using the closed-form solution you derived.
        #  Use only numpy functions. Don't forget regularization!
        
        # w_opt = (X^T X + N * \lambda I)^-1 X^T y
        n = X.shape[1]
        N = X.shape[0]
        
        X_T = np.transpose(X)  # (n, N)
        
        X_T_X = np.matmul(X_T, X) # (n, n)
        
        lambda_I = self.reg_lambda * np.identity(n) * N  # (n, n)
        
        lambda_I[0][0] = 0
                
        X_T_X_lambda_I = X_T_X + lambda_I  # (n, n)
        
        X_T_X_lambda_I_invers = np.linalg.inv(X_T_X_lambda_I)
        
        X_T_y = np.matmul(X_T, y) #(n, 1)

        w_opt = np.matmul(X_T_X_lambda_I_invers, X_T_y)
        
        w_opt = w_opt.reshape(-1)
        
        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


def fit_predict_dataframe(
    model, df: DataFrame, target_name: str, feature_names: List[str] = None,
):
    """
    Calculates model predictions on a dataframe, optionally with only a subset of
    the features (columns).
    :param model: An sklearn model. Must implement fit_predict().
    :param df: A dataframe. Columns are assumed to be features. One of the columns
        should be the target variable.
    :param target_name: Name of target variable.
    :param feature_names: Names of features to use. Can be None, in which case all
        features are used.
    :return: A vector of predictions, y_pred.
    """
    # TODO: Implement according to the docstring description.
    # ====== YOUR CODE: ======
    if not feature_names:
        feature_names = df.columns.tolist()
        feature_names.remove(target_name)
    X = df[feature_names].to_numpy()
    y = df[target_name].to_numpy()
    y_pred = model.fit_predict(X, y)
    # ========================
    return y_pred


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        # TODO:
        #  Add bias term to X as the first feature.
        #  See np.hstack().
        N = X.shape[0]
        B = np.zeros(N) + 1
        B = B.reshape((N,1))
        xb = np.concatenate((B,X),1)

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2, TAX_threshold = 600, INDUS_threshold = 10):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        self.TAX_threshold = TAX_threshold
        self.INDUS_threshold = INDUS_threshold
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        # TODO:
        #  Transform the features of X into new features in X_transformed
        #  Note: You CAN count on the order of features in the Boston dataset
        #  (this class is "Boston-specific"). For example X[:,1] is the second
        #  feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        features_to_polynomial = ['LSTAT', 'RM', 'PTRATIO', 'INDUS', 'TAX', 'NOX', 'CRIM', 'RAD', 'AGE', 'ZN', 'B', 'DIS']
        # features_to_split_linear = ['INDUS', 'TAX']
        all_featres = ["column of bias b", 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
                
        indices_to_polynomial = [all_featres.index(name) for name in features_to_polynomial]
        # indices_to_split_linear = [all_featres.index(name) for name in features_to_split_linear]
        
        columns_to_polynomial = X[:,indices_to_polynomial]
        poly = PolynomialFeatures(self.degree)
        colomns_poly = poly.fit_transform(columns_to_polynomial)
        
        indus_idx = all_featres.index("INDUS")
        indus_column = X[:,indus_idx]
        indus_column = indus_column.reshape(-1,1)
        indus_filtered_to_small = indus_column * (indus_column<self.INDUS_threshold)
        indus_filtered_to_large = indus_column * (indus_column>=self.INDUS_threshold)
        split_indus = np.append(indus_filtered_to_small, indus_filtered_to_large, axis = 1)
        
        tax_idx = all_featres.index("TAX")
        tax_column = X[:,tax_idx]
        tax_column = tax_column.reshape(-1,1)
        tax_filtered_to_small = tax_column * (tax_column<self.TAX_threshold)
        tax_filtered_to_large = tax_column * (tax_column>=self.TAX_threshold)
        split_tax = np.append(tax_filtered_to_small, tax_filtered_to_large, axis = 1)
        
        lstat_idx = all_featres.index("LSTAT")
        lstat_column = X[:,lstat_idx]
        lstat_column = lstat_column.reshape(-1,1)
        
        split_lstat = np.log(lstat_column)
        
        split_features = np.append(split_indus, split_tax, axis = 1)
        split_features = np.append(split_features, split_lstat, axis = 1)
        
        X_transformed = np.append(colomns_poly, split_features, axis = 1)
                
        # ========================
        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    
    corrs_dict = dict()
    
    feature_names = list(df.columns)
    feature_names.remove(target_feature)
    
    taget_data = df[target_feature]
    target_mean = taget_data.mean()
    target_dif = taget_data - target_mean
    target_std = (target_dif**2).sum() ** 0.5
    
    for feature_name in feature_names:
        feature_data = df[feature_name]
        feature_mean = feature_data.mean()
        feature_dif = feature_data - feature_mean
        feature_std = (feature_dif**2).sum() ** 0.5
        
        covarience = (feature_dif*target_dif).sum()
        
        corr = covarience / (target_std * feature_std)
        
        corrs_dict[feature_name] = corr
    
    feature_names.sort(key=lambda x: 1-abs(corrs_dict[x]))
    
    top_n_features = feature_names[:n]
    top_n_corr = [corrs_dict[feature] for feature in top_n_features]
    # ========================

    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """

    # TODO: Implement MSE using numpy.
    # ====== YOUR CODE: ======
    
    mse = ((y-y_pred)**2).mean()
    
    # ========================
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """

    # TODO: Implement R^2 using numpy.
    # ====== YOUR CODE: ======
    y_mean = y.mean()
    denominator = ((y-y_mean)**2).mean()
    numerator = mse_score(y, y_pred)
    r2 = 1 - numerator / denominator
    # ========================
    return r2

from sklearn.model_selection import KFold

def cv_best_hyperparams(
    model: BaseEstimator, X, y, k_folds, degree_range, lambda_range, INDUS_threshold_range = range(5,15), TAX_threshold_range = range(100,800,100)):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities
    #    (recommended). See the docs for the sklearn.model_selection package
    #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree)
    #    you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has
    #    and their names. The parameters dict you return should use the same
    #    names as keys.
    #  - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    kfold = KFold(n_splits=k_folds)
    
    params_to_find = {'linearregressor__reg_lambda': lambda_range,
                      'bostonfeaturestransformer__degree': degree_range,
                      
                      'bostonfeaturestransformer__INDUS_threshold': INDUS_threshold_range,
                      'bostonfeaturestransformer__TAX_threshold': TAX_threshold_range}
    
    feature_names = params_to_find.keys()
    
    best_params = dict()
    
    for feature in feature_names:
        mses = []
        for value in params_to_find[feature]:
            total_mse = 0
            for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
                train_X = X[train_index]
                train_y = y[train_index]
                
                set_dict = {feature: value}
                model.set_params(**set_dict)
                model.fit(train_X, train_y)
                
                test_X = X[test_index]
                test_y = y[test_index]
                pred_y = model.predict(test_X)
                mse = mse_score(test_y, pred_y)
                
                total_mse+=mse
            total_mse/=k_folds
            mses.append(total_mse)
        best_mse_index = min(enumerate(mses), key=lambda x: x[1])[0]
        best_mse_value = params_to_find[feature][best_mse_index]
        best_params[feature] = best_mse_value
        set_dict = {feature: best_mse_value}
        model.set_params(**set_dict)
        
        
    # ========================

    return best_params

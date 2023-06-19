# Import functions from other modules ============================
from io_funs import LoadSave

# Import python libraries ========================================

# Dataset manipulation libraries
import pandas as pd
import numpy as np

from os.path import exists
from time import time
import os

# ML algorithm libraries
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Optimization libraries
from scipy import stats
import skopt
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective
from skopt.plots import plot_evaluations


# Data Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

from bokeh.io import output_notebook
from bokeh.layouts import row, column
output_notebook()
from bokeh.plotting import show,figure
TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
]
from skopt.plots import plot_evaluations

# check later if you need this
# from bokeh.palettes import Category20, colorblind
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import warnings
# warnings.filterwarnings('ignore')


# ===============================================================================
# ==================                                           ==================
# ==================      Class Train  Ml  Regression          ==================
# ==================                                           ==================
# ===============================================================================


class TrainMlRegression:
    """
    perform traditional ML training

    Tasks
    ------
    - Process dataset: Scale, split_train_val_test
    - Train ML model
    - Optimize
    - Feature reduction: PCA, RFE
    - Output: Save trained models, metrics

    Attributes
    -----------
    - feature_values: array
            flux arrays in XXXX [write the dimension]
    - feature_names: list
            name of wavelength in micron
    - target_values: array
            target variable array: Temperature, Gravity, Carbon_to_Oxygen, Metallicity
    - is_tuned: str
            To optimize the hyperparameters: 'yes' or 'no'
    - param_grid: dic
            ML hyperparameters to be tuned.
            Note: param_grid is used if is_tuned = 'yes'
    - spectral_resolution: int
            resolution of the synthetic spectra used to generate the dataset
    - is_feature_improved: str
        options:
            'no': all features
            'pca': used Principal component analysis method for dimensionality reduction
            'RFE': used Recursive Feature Elimination (RFE) for Feature Selection
    - n_jobs: int
            number of processors for optimization step
    - cv: int
            Cross Validation
    - is_augmented: str
        options:
            'no': used native dataset
            [METHOD]: augmented dataset like adding noise etc.
    - ml_model: object from sklearn package
            Use the library name to get instintiated e.g. xgboost()
    - ml_model_str: str
            name of the ML model

    Outputs
    --------
    - Trained ML models

    """

    def __init__(self,
                 feature_values,
                 feature_names,
                 target_values,
                 target_name,
                 is_tuned,
                 param_grid,
                 spectral_resolution,
                 is_feature_improved,
                 n_jobs,
                 cv,
                 is_augmented,
                 ml_model,
                 ml_model_str,
                 ):

        self.feature_values = feature_values
        self.feature_names = feature_names
        self.target_values = target_values
        self.target_name = target_name
        self.is_tuned = is_tuned
        self.param_grid = param_grid
        self.spectral_resolution = spectral_resolution
        # self.wl = wl
        self.feature_names = feature_names
        self.is_feature_improved = is_feature_improved

        self.n_jobs = n_jobs
        self.cv = cv
        self.is_augmented = is_augmented
        self.ml_model = ml_model
        self.ml_model_str = ml_model_str



    def split_train_test(self,
                         test_size = 0.1):
        """
        split the loaded set into train and test sets

        Inputs
        ------
            - self.feature_values
            - self.target_values
            - test_size: default is 10% kept for testing the final trained model
            - shuffle: default is 'True'
            - random_state: default is 42
        Return
        -------
            - self.X_train, self.X_test: Used to train the machine learning model and evaluate it later.
            - self.y_train, self.y_test: Targets used for train/test the models

        Refs
        -----
            - SciKit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.feature_values,
                                                                                self.target_values,
                                                                                test_size = 0.10,
                                                                                shuffle = True,
                                                                                random_state = 42,
                                                                            )



    def split_train_validation_test(self, test_size=0.1, val_size=0.1):
        """
        Split the loaded set into train, validation, and test sets

        Inputs:
        - test_size: float
            Proportion of the dataset to include in the test split
        - val_size: float
            Proportion of the remaining train dataset to include in the validation split

        Returns:
        - self.X_train, self.X_val, self.X_test: arrays
            Used to train, validate, and evaluate the machine learning model, respectively
        - self.y_train, self.y_val, self.y_test: arrays
            Targets used for training, validation, and testing the models

        References:
        - SciKit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.feature_values,
            self.target_values,
            test_size=test_size,
            shuffle=True,
            random_state=42
        )
        self.X_test, self.y_test = X_test, y_test

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            shuffle=True,
            random_state=42
        )



    def normalize_X_column_wise(self, X_train=None, X_val=None, X_test=None, print_model=False):
        """
        Normalize features/column variables to a specified range.
        Transform your data such that its values are within the specified range [0, 1].

        Inputs:
            - X_train (numpy array): Training feature matrix
            - X_val (numpy array): Validation feature matrix
            - X_test (numpy array): Test feature matrix
            - print_model (bool): Whether to print the trained normalizer model

        Assigns:
            - self.X_train_normalized_columnwise (numpy array): Normalized training feature matrix
            - self.X_val_normalized_columnwise (numpy array): Normalized validation feature matrix
            - self.X_test_normalized_columnwise (numpy array): Normalized test feature matrix
            - self.normalize_X_ColumnWise (MinMaxScaler): Trained normalizer object
        """
        # Set default values if None is provided
        X_train = self.X_train if X_train is None else X_train
        X_val = self.X_val if X_val is None else X_val
        X_test = self.X_test if X_test is None else X_test

        normalizer = MinMaxScaler(feature_range=(0, 1))
        self.X_train_normalized_columnwise = normalizer.fit_transform(X_train)
        self.X_val_normalized_columnwise = normalizer.transform(X_val)
        self.X_test_normalized_columnwise = normalizer.transform(X_test)
        self.normalize_X_ColumnWise = normalizer

        LoadSave(self.ml_model_str, self.is_feature_improved, self.is_augmented, self.is_tuned).load_or_dump_trained_object(
            trained_object=self.normalize_X_ColumnWise, indicator='normalize_X_ColumnWise', load_or_dump='dump')

        if print_model:
            print(normalizer)
            
            
            
            
            
            

    def normalize_y_column_wise(self, y_train=None, y_val=None, y_test=None, print_model=False):
        """
        Scale target variable (y) and column-wise feature variables to a specified range.
        Transform the data such that its values are within the specified range [0, 1].

        Inputs:
            - y_train (numpy array): Training target variable array
            - y_val (numpy array): Validation target variable array
            - y_test (numpy array): Test target variable array
            - print_model (bool): Whether to print the trained scaler model

        Assigns:
            - self.y_train_normalized_columnwise (numpy array): Scaled training target variable array
            - self.y_val_normalized_columnwise (numpy array): Scaled validation target variable array
            - self.y_test_normalized_columnwise (numpy array): Scaled test target variable array
        """
        # Set default values if None is provided
        y_train = self.y_train if y_train is None else y_train
        y_val = self.y_val if y_val is None else y_val
        y_test = self.y_test if y_test is None else y_test

        scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.y_train_normalized_columnwise = scaler_y.fit_transform(y_train)
        self.y_val_normalized_columnwise = scaler_y.transform(y_val)
        self.y_test_normalized_columnwise = scaler_y.transform(y_test)
        self.normalize_y_ColumnWise = scaler_y

        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned).load_or_dump_trained_object(
                     trained_object=self.normalize_y_ColumnWise,
                     indicator='normalize_y_ColumnWise',
                     load_or_dump='dump')

        if print_model:
            print(scaler_y)





    def standardize_y_column_wise(self, y_train=None, y_val=None, y_test=None, print_model=False):
        """
        Standardize target variable (y) column-wise by removing the mean and scaling to unit variance.
        Transform the data such that its distribution will have a mean value of 0 and a standard deviation of 1.

        Inputs:
            - y_train (numpy array): Training target variable array
            - y_val (numpy array): Validation target variable array
            - y_test (numpy array): Test target variable array
            - print_model (bool): Whether to print the trained scaler model

        Assigns:
            - self.y_train_standardized_columnwise (numpy array): Standardized training target variable array
            - self.y_val_standardized_columnwise (numpy array): Standardized validation target variable array
            - self.y_test_standardized_columnwise (numpy array): Standardized test target variable array
        """
        # Set default values if None is provided
        y_train = self.y_train if y_train is None else y_train
        y_val = self.y_val if y_val is None else y_val
        y_test = self.y_test if y_test is None else y_test
        
        scaler_y = StandardScaler()
        self.y_train_standardized_columnwise = scaler_y.fit_transform(y_train)
        self.y_val_standardized_columnwise = scaler_y.transform(y_val)
        self.y_test_standardized_columnwise = scaler_y.transform(y_test)
        self.standardize_y_ColumnWise = scaler_y

        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned).load_or_dump_trained_object(
                     trained_object=self.standardize_y_ColumnWise,
                     indicator='standardize_y_ColumnWise',
                     load_or_dump='dump')

        if print_model:
            print(scaler_y)


    

    def standardize_X_column_wise(self, X_train=None, X_val=None, X_test=None, print_model=False):
        """
        Standardize feature variables (X) column-wise by removing the mean and scaling to unit variance.
        Transform the data such that each feature will have a mean value of 0 and a standard deviation of 1.

        Inputs:
            - X_train (numpy array): Training feature matrix
            - X_val (numpy array): Validation feature matrix
            - X_test (numpy array): Test feature matrix
            - print_model (bool): Whether to print the trained scaler model

        Assigns:
            - self.X_train_standardized_columnwise (numpy array): Standardized training feature matrix
            - self.X_val_standardized_columnwise (numpy array): Standardized validation feature matrix
            - self.X_test_standardized_columnwise (numpy array): Standardized test feature matrix
        """
        # Set default values if None is provided
        X_train = self.X_train if X_train is None else X_train
        X_val = self.X_val if X_val is None else X_val
        X_test = self.X_test if X_test is None else X_test
        
        scaler_X = StandardScaler()
        # if X_train == None:
        self.X_train_standardized_columnwise = scaler_X.fit_transform(X_train)
        self.X_val_standardized_columnwise = scaler_X.transform(X_val)
        self.X_test_standardized_columnwise = scaler_X.transform(X_test)
        # elif X_train:
        X_train_standardized_columnwise = scaler_X.fit_transform(X_train)
        X_val_standardized_columnwise = scaler_X.transform(X_val)
        X_test_standardized_columnwise = scaler_X.transform(X_test)
        
        
        
        self.standardize_X_ColumnWise = scaler_X

        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned).load_or_dump_trained_object(
                     trained_object=self.standardize_X_ColumnWise,
                     indicator='standardize_X_ColumnWise',
                     load_or_dump='dump')

        if print_model:
            print(scaler_X)



    def standardize_X_row_wise(self, X_train=None, X_val=None, X_test=None, print_model=False):
        """
        Standardize feature variables (X) column-wise by removing the mean and scaling to unit variance.
        Transform the data such that each feature will have a mean value of 0 and a standard deviation of 1.

        Inputs:
            - X_train (numpy array): Training feature matrix
            - X_val (numpy array): Validation feature matrix
            - X_test (numpy array): Test feature matrix
            - print_model (bool): Whether to print the trained scaler model

        Assigns:
            - self.X_train_standardized_columnwise (numpy array): Standardized training feature matrix
            - self.X_val_standardized_columnwise (numpy array): Standardized validation feature matrix
            - self.X_test_standardized_columnwise (numpy array): Standardized test feature matrix
        """
        # Set default values if None is provided
        X_train = self.X_train if X_train is None else X_train
        X_val = self.X_val if X_val is None else X_val
        X_test = self.X_test if X_test is None else X_test
        
        scaler_X = StandardScaler()
        self.X_train_standardized_rowwise = scaler_X.fit_transform(X_train.T).T
        self.X_val_standardized_rowwise = scaler_X.fit_transform(X_val.T).T
        self.X_test_standardized_rowwise = scaler_X.fit_transform(X_test.T).T
        self.standardize_X_RowWise = scaler_X

        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned).load_or_dump_trained_object(
                     trained_object=self.standardize_X_RowWise,
                     indicator='standardize_X_RowWise',
                     load_or_dump='dump')

        if print_model:
            print(scaler_X)
                        


    def train_regression_model(self,
                               trained_model = None,
                               X_train = None,
                               X_val = None,
                               X_test = None,

                               y_train = None,
                               y_val = None,
                               y_test = None,
                               
                               is_tuned = 'no',
                               n_iter = 3,  # number of iterations for Bayesian Optimization
                               verbose = 1, # print output
                               plot_results = True,
                               print_results = True,
                               ):
        """
        Train ML regression model using traditional ML algorithms using BayesSearchCV

        Inputs
        -------
            -  self.is_tuned
            -  self.ml_model
            -  self.cv
            -  self.param_grid
            -  self.n_jobs
            -  verbose = 1
            -  scoring = mean_squared_error # Mean squared error regression loss.
        Returns
        --------
            - Trained ML model
        """
        # Set default values if None is provided
        X_train = self.X_train if X_train is None else X_train
        X_val = self.X_val if X_val is None else X_val
        X_test = self.X_test if X_test is None else X_test
        
        y_train = self.y_train if y_train is None else y_train
        y_val = self.y_val if y_val is None else y_val
        y_test = self.y_test if y_test is None else y_test
        
        
        if self.is_tuned == 'yes':

            model = BayesSearchCV(self.ml_model,
                                  search_spaces = self.param_grid,
                                  n_iter = n_iter,
                                  cv = self.cv,
                                  n_jobs = self.n_jobs,
                                  verbose = 2,
                                  )

            model.fit( X_train, y_train)

            self.optimized_params = {}
            self.optimized_params = model.best_params_
            self.trained_model = model


            if plot_results:
                assert isinstance( model.optimizer_results_, object)
                plot_evaluations( model.optimizer_results_[0],
                                 # bins=10,
                                 dimensions=[xx.replace('estimator__', '') for xx in list(self.param_grid.keys())]
                                 )

            if print_results:
                print(' ==============    Optimal HyperParameters    ============== ')
                # display(self.optimized_params)
                print("total_iterations", self.trained_model.total_iterations)
                print("val. score: %s" % self.trained_model.best_score_)
                print("test score: %s" % self.trained_model.score(self.X_test, self.y_test))
                display("best params: %s" % str(self.trained_model.best_params_))
                

        if self.is_tuned == 'no':
            # Instantiate the ML model using default parameters with NO evaluation set
            model = self.ml_model
            model.fit(X_train, y_train)
            self.optimized_params = model.get_params(deep=True)
            self.trained_model = model

            if print_results:
                print(' ==============    Optimal HyperParameters    ============== ')
                # display(self.optimized_params)
                print("val. score: %s" % self.trained_model.best_score_)
                print("test score: %s" % self.trained_model.score(self.X_test, self.y_test))
                display("best params: %s" % str(self.trained_model.best_params_))
                

        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned, ).load_or_dump_trained_object(trained_object=model,
                                                              indicator='TrainedModel',
                                                              load_or_dump='dump')
        
        
        
        
        

    def plot_boxplot_scaled_features(self, scaled_feature, title = None):
        """
        Interpretation: 
        - Median: middle quartile marks
        - Inter-quartile range: (The middle “box”): 50% of scores fall within the inter-quartile range
        - Upper quartile: 75% of the scores fall below the upper quartile.
        - Lower quartile: 25% of scores fall below the lower quartile.
        """
        plt.figure(figsize=(12, 5))
        plt.boxplot(scaled_feature, sym='')
        
        if len(scaled_feature) > 10:
            plt.xticks(rotation=90)

        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Scaled Value', fontsize=12)
        if title: 
            plt.title(title, fontsize=14)
            
        plt.tight_layout()
        plt.show()


    def plot_histogram_scaled_features(self, scaled_feature):
        # Plotting the histogram
        plt.figure(figsize=(15, 6))
        plt.hist(scaled_feature, bins=31)
        plt.xlabel('Scaled Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Scaled Features')
        plt.show()
        

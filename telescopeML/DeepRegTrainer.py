# Import functions from other modules ============================
from io_funs import LoadSave

# Import python libraries ========================================
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
#
# from tensorflow.keras.models import save_model
# import pickle as pk
# from tensorflow.keras.models import load_model

# Import BOHB Package ========================================

# Libraries for BOHB Package 
import logging
logging.basicConfig(level=logging.WARNING)

# import argparse
#
# import hpbandster.core.nameserver as hpns
# import hpbandster.core.result as hpres
#
# from hpbandster.optimizers import BOHB as BOHB
# from hpbandster.examples.commons import MyWorker
#
# from tensorflow.keras.models import load_model
# import ConfigSpace as CS
# from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.examples.commons import MyWorker

from tensorflow.keras.models import load_model


from bokeh.io import output_notebook
from bokeh.layouts import row, column
output_notebook()
from bokeh.plotting import show,figure
TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
]
# ===============================================================================
# ==================                                           ==================
# ==================            Train CNN Regression           ==================
# ==================                                           ==================
# ===============================================================================  

from typing import List, Union, Dict
from sklearn.base import BaseEstimator


class TrainRegression:
    """
    Perform Convolutional Neural Network training

    Tasks:
    - Process dataset: Scale, split_train_val_test
    - Train CNN model
    - Optimize
    - Feature reduction: PCA, RFE
    - Output: Save trained models, metrics

    Attributes:
    - trained_model: object
        Trained ML model (optional)
    - trained_model_history: dict
        History dict from the trained model 
    - feature_values: array
        Flux arrays (input data)
    - feature_names: list
        Name of wavelength in micron
    - target_values: array
        Target variable array (e.g., Temperature, Gravity, Carbon_to_Oxygen, Metallicity)
    - target_name: str
        Name of the target variable
    - is_hyperparam_tuned: str
        Indicates whether hyperparameters are tuned or not ('yes' or 'no')
    - param_grid: dict
        ML hyperparameters to be tuned (used if is_hyperparam_tuned = 'yes')
    - spectral_resolution: int
        Resolution of the synthetic spectra used to generate the dataset
    - feature_improvement_method: str
        Indicates the method used for feature improvement ('no', 'pca', 'RFE')
    - n_jobs: int
        Number of processors for optimization step
    - cv: int
        Cross-validation
    - augmentation_method: str
        Indicates if augmented dataset is used ('no' or method name)
    - ml_model: object
        ML model object from sklearn package
    - ml_model_str: str
        Name of the ML model

    Outputs:
    - Trained ML models
    """

    def __init__(
        self,
        trained_model: Union[None, BaseEstimator] = None,
        trained_model_history: Union[None, Dict] = None,
        feature_values: Union[None, np.ndarray] = None,
        feature_names: Union[None, List[str]] = None,
        target_values: Union[None, np.ndarray] = None,
        target_name: Union[None, str] = None,
        is_tuned: str = 'no',
        param_grid: Union[None, Dict] = None,
        spectral_resolution: Union[None, int] = None,
        is_feature_improved: str = 'no',
        n_jobs: Union[None, int] = None,
        cv: Union[None, int] = None,
        is_augmented: str = 'no',
        ml_model: Union[None, BaseEstimator] = None,
        ml_model_str: Union[None, str] = None,
    ) -> None:

        self.trained_model = trained_model
        self.trained_model_history = trained_model_history
        self.feature_values = feature_values
        self.feature_names = feature_names
        self.target_values = target_values
        self.target_name = target_name
        self.is_tuned = is_tuned
        self.param_grid = param_grid
        self.spectral_resolution = spectral_resolution
        self.is_feature_improved = is_feature_improved
        self.n_jobs = n_jobs
        self.cv = cv
        self.is_augmented = is_augmented
        self.ml_model = ml_model
        self.ml_model_str = ml_model_str


    def split_train_test(self, test_size=0.1):
        """
        Split the loaded set into train and test sets

        Inputs:
        - test_size: float
            The proportion of the dataset to include in the test split

        Returns:
        - X_train, X_test, y_train, y_test: arrays
            Train and test datasets for features and targets

        References:
        - SciKit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.feature_values,
                                                            self.target_values,
                                                            test_size=test_size,
                                                            shuffle=True,
                                                            random_state=42)

        
        
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



    def normalize_X_row_wise(self, X_train=None, X_val=None, X_test=None, print_model=False):
        """
        Normalize observations/instances/row variables to a specified range.
        Transform your data such that its values are within the specified range [0, 1].

        Inputs:
            - X_train (numpy array): Training feature matrix
            - X_val (numpy array): Validation feature matrix
            - X_test (numpy array): Test feature matrix
            - print_model (bool): Whether to print the trained normalizer model

        Assigns:
            - self.X_train_normalized_rowwise (numpy array): Normalized training feature matrix
            - self.X_val_normalized_rowwise (numpy array): Normalized validation feature matrix
            - self.X_test_normalized_rowwise (numpy array): Normalized test feature matrix
        """
        # Set default values if None is provided
        X_train = self.X_train if X_train is None else X_train
        X_val = self.X_val if X_val is None else X_val
        X_test = self.X_test if X_test is None else X_test
        
        normalizer = MinMaxScaler(feature_range=(0, 1))
        self.X_train_normalized_rowwise = normalizer.fit_transform(X_train.T).T
        self.X_val_normalized_rowwise = normalizer.fit_transform(X_val.T).T
        self.X_test_normalized_rowwise = normalizer.fit_transform(X_test.T).T
        self.normalize_X_RowWise = normalizer

        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned).load_or_dump_trained_object(
                     trained_object=self.normalize_X_RowWise,
                     indicator='normalize_X_RowWise',
                     load_or_dump='dump')

        if print_model:
            print(normalizer)



        # Apply scaling using the global mean and standard deviation to the train dataset
        self.X_train_norm    def normalize_X_global_column_wise(self, X_train=None, X_val=None, X_test=None, print_model=False):

        # Set default values if None is provided
        X_train = self.X_train if X_train is None else X_train
        X_val = self.X_val if X_val is None else X_val
        X_test = self.X_test if X_test is None else X_test

        # Concatenate the train, validation, and test datasets
        all_data = pd.concat([X_train, X_val, X_test], axis=0)

        # Calculate the global mean and standard deviation of X
        global_min = all_data.min()
        global_max = all_data.max()
alized_columnwise_global = (train_df - global_min) / (global_max - global_min)

        # Apply scaling using the global mean and standard deviation to the validation dataset
        self.X_val_normalized_columnwise_global = (val_df - global_min) / (global_max - global_min)

        # Apply scaling using the global mean and standard deviation to the test dataset
        self.X_test_normalized_columnwise_global = (test_df - global_min) / (global_max - global_min)

        if print_model:
            print("global_min, global_max: ", global_min, global_max)    
    

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
            
            
            

    def standardize_X_row_column_wise(self, X_train=None, X_val=None, X_test=None, print_model=False):
        """
        Standardize feature variables (X) row-wise and column-wise by removing the mean and scaling to unit variance.
        Transform the data such that each row and column will have a mean value of 0 and a standard deviation of 1.

        Inputs:
            - X_train (numpy array): Training feature matrix
            - X_val (numpy array): Validation feature matrix
            - X_test (numpy array): Test feature matrix
            - print_model (bool): Whether to print the trained scaler model

        Assigns:
            - self.X_train_standardized_rowcolumn (numpy array): Standardized training feature matrix
            - self.X_val_standardized_rowcolumn (numpy array): Standardized validation feature matrix
            - self.X_test_standardized_rowcolumn (numpy array): Standardized test feature matrix
        """
        # Set default values if None is provided
        X_train = self.X_train if X_train is None else X_train
        X_val = self.X_val if X_val is None else X_val
        X_test = self.X_test if X_test is None else X_test
        
        scaler_row = StandardScaler()
        X_train_standardized_rowwise = scaler_row.fit_transform(X_train.T).T
        X_val_standardized_rowwise = scaler_row.fit_transform(X_val.T).T
        X_test_standardized_rowwise = scaler_row.fit_transform(X_test.T).T

        scaler_column = StandardScaler()
        # X_train_standardized_rowcolumn = scaler_column.fit_transform(X_train_standardized_rowwise)
        # X_val_standardized_rowcolumn = scaler_column.transform(X_val_standardized_rowwise)
        # X_test_standardized_rowcolumn = scaler_column.transform(X_test_standardized_rowwise)

        self.X_train_standardized_rowcolumn = X_train_standardized_rowcolumn
        self.X_val_standardized_rowcolumn = X_val_standardized_rowcolumn
        self.X_test_standardized_rowcolumn = X_test_standardized_rowcolumn

        self.standardize_X_RowColumnWise = (scaler_row, scaler_column)

        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned).load_or_dump_trained_object(
                     trained_object=self.standardize_X_RowColumnWise,
                     indicator='standardize_X_RowColumnWise',
                     load_or_dump='dump')

        if print_model:
            print(scaler_row)
            print(scaler_column)

            
            

    def normalize_X_row_column_Wise(self, X_train=None, X_val=None, X_test=None, print_model=False):
        """
        Normalize feature variables (X) row-wise and column-wise using Min-Max normalization.
        Transform the data such that each row and column values are within the specified range [0, 1].

        Inputs:
            - X_train (numpy array): Training feature matrix
            - X_val (numpy array): Validation feature matrix
            - X_test (numpy array): Test feature matrix
            - print_model (bool): Whether to print the trained scaler model

        Assigns:
            - self.X_train_normalized_rowcolumn (numpy array): Normalized training feature matrix
            - self.X_val_normalized_rowcolumn (numpy array): Normalized validation feature matrix
            - self.X_test_normalized_rowcolumn (numpy array): Normalized test feature matrix
        """
        # Set default values if None is provided
        X_train = self.X_train if X_train is None else X_train
        X_val = self.X_val if X_val is None else X_val
        X_test = self.X_test if X_test is None else X_test
        
        scaler_row = MinMaxScaler(feature_range=(0, 1))
        X_train_normalized_rowwise = scaler_row.fit_transform(X_train.T).T
        X_val_normalized_rowwise = scaler_row.transform(X_val.T).T
        X_test_normalized_rowwise = scaler_row.transform(X_test.T).T

        scaler_column = MinMaxScaler(feature_range=(0, 1))
        X_train_normalized_rowcolumn = scaler_column.fit_transform(X_train_normalized_rowwise)
        X_val_normalized_rowcolumn = scaler_column.transform(X_val_normalized_rowwise)
        X_test_normalized_rowcolumn = scaler_column.transform(X_test_normalized_rowwise)

        self.X_train_normalized_rowcolumn = X_train_normalized_rowcolumn
        self.X_val_normalized_rowcolumn = X_val_normalized_rowcolumn
        self.X_test_normalized_rowcolumn = X_test_normalized_rowcolumn

        self.MinMaxScaler_X_RowColumnWise = (scaler_row, scaler_column)

        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned).load_or_dump_trained_object(
                     trained_object=self.MinMaxScaler_X_RowColumnWise,
                     indicator='MinMaxScaler_X_RowColumnWise',
                     load_or_dump='dump')

        if print_model:
            print(scaler_row)
            print(scaler_column)
  

    def plot_boxplot_scaled_features(self, scaled_feature, title = None, xticks_list = None):
        """
        Interpretation: 
        - Median: middle quartile marks
        - Inter-quartile range: (The middle “box”): 50% of scores fall within the inter-quartile range
        - Upper quartile: 75% of the scores fall below the upper quartile.
        - Lower quartile: 25% of scores fall below the lower quartile.
        """
        plt.figure(figsize=(12, 3))
        plt.boxplot(scaled_feature, sym='')
        
        if len(scaled_feature) > 10:
            plt.xticks(rotation=45)

        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Scaled Value', fontsize=12)
        if title: 
            plt.title(title, fontsize=14)
            
        # Add custom x-ticks
        # custom_xticks = ['Label 1', 'Label 2', 'Label 3', 'Label 4']
        if xticks_list:
            plt.xticks(xticks_list)

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
        
    def plot_model_loss (self, history = None, title = None):
        
        # from bokeh.plotting import figure, show
        # from bokeh.models import Legend
        
        history = self.trained_model_history if history is None else history
        # Define the epochs as a list
        epochs = list(range(len(history.history['loss'])))

        # Define colorblind-friendly colors
        colors = ['#d62728',  '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']

        # Create a new figure
        p = figure(title=title , width=1000, height=300, y_axis_type='log', x_axis_label='Epochs', y_axis_label='Loss')

        # Add the data lines to the figure with colorblind-friendly colors and increased line width
        p.line(epochs, history.history['loss'], line_color=colors[0], line_dash='solid', line_width=2, legend_label='Total loss')
        p.line(epochs, history.history['val_loss'], line_color=colors[0], line_dash='dotted', line_width=2)

        p.line(epochs, history.history['gravity_loss'], line_color=colors[1], line_dash='solid', line_width=2, legend_label='gravity')
        p.line(epochs, history.history['val_gravity_loss'], line_color=colors[1], line_dash='dotted', line_width=2)

        p.line(epochs, history.history['c_o_ratio_loss'], line_color=colors[2], line_dash='solid', line_width=2, legend_label='c_o_ratio')
        p.line(epochs, history.history['val_c_o_ratio_loss'], line_color=colors[2], line_dash='dotted', line_width=2)

        p.line(epochs, history.history['metallicity_loss'], line_color=colors[3], line_dash='solid', line_width=2, legend_label='metallicity')
        p.line(epochs, history.history['val_metallicity_loss'], line_color=colors[3], line_dash='dotted', line_width=2)

        p.line(epochs, history.history['temperature_loss'], line_color=colors[4], line_dash='solid', line_width=2, legend_label='temperature')
        p.line(epochs, history.history['val_temperature_loss'], line_color=colors[4], line_dash='dotted', line_width=2)

        # Increase size of x and y ticks
        p.title.text_font_size = '14pt'
        p.xaxis.major_label_text_font_size = '12pt'
        p.xaxis.axis_label_text_font_size = '12pt'
        p.yaxis.major_label_text_font_size = '12pt'
        p.yaxis.axis_label_text_font_size = '12pt'

        # display legend in top left corner (default is top right corner)
        p.legend.location = "bottom_left"


        # change appearance of legend text
        # p.legend.label_text_font = "times"
        # p.legend.label_text_font_style = "italic"
        # p.legend.label_text_color = "navy"

        # change border and background of legend
        # p.legend.border_line_width = 3
        # p.legend.border_line_color = "navy"
        # p.legend.border_line_alpha = 0.8
        p.legend.background_fill_color = 'white'
        p.legend.background_fill_alpha = 0.5


        # Show the plot
        show(p)

      

    def train_ml_regression_model(self,
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
        trained_model = self.trained_model if trained_model is None else trained_model
        is_tuned = self.is_tuned if is_tuned is None else is_tuned

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
    




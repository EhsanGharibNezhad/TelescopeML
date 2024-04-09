# Import functions from other modules ============================
from .IO_utils import *

# Import python libraries ========================================

# ******* Standard Data Manipulation / Statistical Libraries *****
import pandas as pd

pd.options.mode.chained_assignment = None  # Suppress warnings
import numpy as np
import pickle as pk

from typing import List, Union, Dict
from sklearn.base import BaseEstimator

# ******* Data Visualization Libraries ****************************

import matplotlib.pyplot as plt

from bokeh.io import output_notebook

output_notebook()
from bokeh.plotting import show, figure

TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
]

# ******** Data science / Machine learning Libraries ***************
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf

tf.get_logger().setLevel('ERROR')


# ===============================================================================
# ==================                                           ==================
# ==================            Data Processor                 ==================
# ==================                                           ==================
# ===============================================================================

class DataProcessor:
    """
    Perform various tasks to process the datasets, including:

    - Prepare inputs and outputs
    - Split the dataset into training, validation, and test sets
    - Scale/normalize the data
    - Visualize the data
    - Conduct feature engineering

    Parameters
    ----------
    flux_values : np.ndarray
        Flux arrays (input data).
    wavelength_names : List[str]
        Name of wavelength in micron.
    wavelength_values : np.ndarray
        wavelength array in micron.
    output_values : np.ndarray
        output variable array (e.g., Temperature, Gravity, Carbon_to_Oxygen, Metallicity).
    output_names : List[str]
        Name of the output variable.
    spectral_resolution : int, optional
        Resolution of the synthetic spectra used to generate the dataset.
    trained_ML_model : BaseEstimator, optional
        ML model object from sklearn package.
    trained_ML_model_name : str, optional
        Name of the ML model.
    ml_method : str, optional
        Machine learning method ('regression' or 'classification').

    """
    def __init__(
            self,
            flux_values: Union[np.ndarray] = None,
            wavelength_names: Union[List[str]] = None,
            wavelength_values: Union[np.ndarray] = None,
            output_values: Union[np.ndarray] = None,
            output_names: Union[str] = None,
            spectral_resolution: Union[None, int] = None,
            trained_ML_model: Union[None, BaseEstimator] = None,
            trained_ML_model_name: Union[None, str] = None,
            ml_method: str = 'regression',
    ):

        self.flux_values = flux_values
        self.wavelength_names = wavelength_names
        self.wavelength_values = wavelength_values
        self.output_values = output_values
        self.output_names = output_names
        self.spectral_resolution = spectral_resolution
        self.trained_ML_model = trained_ML_model
        self.trained_ML_model_name = trained_ML_model_name
        self.ml_method = ml_method
        self.LoadSave = LoadSave(trained_ML_model_name,
                                 ml_method,
                                 )

    def split_train_test(self, test_size=0.1):
        """
        Split the loaded dataset into train and test sets

        Parameters
        ----------
        test_size : float
            The proportion of the dataset to include in the test split (default = 0.1).

        Returns
        -------
        self.X_train : array
            X train set.
        self.X_test : array
            X test set.
        self.y_train : array
            y train set.
        self.y_test : array
            y test set.

        References
        ----------
        link: `sklearn.model_selection.train_test_split <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_

        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.flux_values,
                                                                                self.output_values,
                                                                                test_size=test_size,
                                                                                shuffle=True,
                                                                                random_state=42)

    def split_train_validation_test(self,
                                    test_size=0.1,
                                    val_size=0.1,
                                    random_state_=42):
        """
        Split the loaded dataset into train, validation, and test sets.

        Parameters
        ----------
        test_size : float
            Proportion of the dataset to include in the test split (default = 0.1).
        val_size : float
            Proportion of the remaining train dataset to include in the validation split (default = 0.1).

        Returns
        -------
        self.X_train : array
            Used to train the machine learning model.
        self.X_val : array
            Used to validate the machine learning model.
        self.X_test : array
            Used to evaluate the machine learning model.
        self.y_train : array
            Outputs used for training the models.
        self.y_val : array
            Outputs used for validating the models.
        self.y_test : array
            Outputs used for testing the models.

        References
        ----------
        link: `sklearn.model_selection.train_test_split <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_
        """

        X_train, X_test, y_train, y_test = train_test_split(
            self.flux_values,
            self.output_values,
            test_size=test_size,
            shuffle=True,
            random_state=random_state_
        )
        self.X_test, self.y_test = X_test, y_test

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            shuffle=True,
            random_state=random_state_
        )

    def normalize_X_column_wise(self,
                                output_indicator='Trained_Normalizer_X_ColWise',
                                X_train=None,
                                X_val=None,
                                X_test=None,
                                print_model=False):
        """
        Normalize features/column variables to a specified range [0, 1].

        Parameters
        ----------
        X_train : array
            Training feature matrix.
        X_val : array
            Validation feature matrix.
        X_test : array
            Test feature matrix.
        print_model : bool, optional
            Whether to print the trained normalizer model.

        Returns
        -------
        self.X_train_normalized_columnwise : array
            Normalized training feature matrix.
        self.X_val_normalized_columnwise : array
            Normalized validation feature matrix.
        self.X_test_normalized_columnwise : array
            Normalized test feature matrix.
        self.normalize_X_ColumnWise : MinMaxScaler
            Trained normalizer object.

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

        self.LoadSave.load_or_dump_trained_object(
            trained_object=self.normalize_X_ColumnWise,
            output_indicator=output_indicator,
            load_or_dump='dump')

        if print_model:
            print(normalizer)

    def normalize_X_row_wise(self,
                             output_indicator='Trained_Normalizer_X_RowWise',
                             X_train=None,
                             X_val=None,
                             X_test=None,
                             print_model=False):
        """
        Normalize observations/instances/row variables to a specified range [0, 1].

        Transform your data such that its values are within the specified range [0, 1].

        Parameters
        ----------
        X_train : array
            Training feature matrix.
        X_val : array
            Validation feature matrix.
        X_test : array
            Test feature matrix.
        print_model : bool, optional
            Whether to print the trained normalizer model.

        Returns
        -------
        self.X_train_normalized_rowwise : array
            Normalized training feature matrix.
        self.X_val_normalized_rowwise : array
            Normalized validation feature matrix.
        self.X_test_normalized_rowwise : array
            Normalized test feature matrix.
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

        self.LoadSave.load_or_dump_trained_object(
            trained_object=self.normalize_X_RowWise,
            output_indicator=output_indicator,
            load_or_dump='dump')

        if print_model:
            print(normalizer)

    def normalize_y_column_wise(self,
                                output_indicator =  'Trained_Normalizer_y_ColWise',
                                y_train=None,
                                y_val=None, y_test=None, print_model=False):
        """
        Scale target variable (y) and column-wise feature variables to a specified range.
        Transform the data such that its values are within the specified range [0, 1].

        Parameters
        -----------
        y_train : array
            Training target variable array.
        y_val : array
            Validation target variable array.
        y_test : array
            Test target variable array.
        print_model : bool
            Whether to print the trained scaler model.

        Returns
        -------
        self.y_train_normalized_columnwise : array
            Scaled training target variable array.
        self.y_val_normalized_columnwise : array
            Scaled validation target variable array.
        self.y_test_normalized_columnwise : array
            Scaled test target variable array.
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

        self.LoadSave.load_or_dump_trained_object(
            trained_object=self.normalize_y_ColumnWise,
            output_indicator=output_indicator,
            load_or_dump='dump')

        if print_model:
            print(scaler_y)

    def standardize_y_column_wise(self,
                                  output_indicator = 'Trained_StandardScaler_y_ColWise',
                                  y_train=None,
                                  y_val=None,
                                  y_test=None,
                                  print_model=False):
        """
        Standardize target variable (y) column-wise by removing the mean and scaling to unit variance.
        Transform the data such that its distribution will have a mean value of 0 and a standard deviation of 1.

        Parameters
        -----------
        y_train : array
            Training target variable array.
        y_val : array
            Validation target variable array.
        y_test : array
            Test target variable array.
        print_model : bool
            Whether to print the trained scaler model.

        Returns
        -------
        self.y_train_standardized_columnwise : array
            Standardized training target variable array.
        self.y_val_standardized_columnwise : array
            Standardized validation target variable array.
        self.y_test_standardized_columnwise : array
            Standardized test target variable array.
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

        self.LoadSave.load_or_dump_trained_object(
                                trained_object=self.standardize_y_ColumnWise,
                                output_indicator=output_indicator,
                                load_or_dump='dump')

        if print_model:
            print(scaler_y)


    def standardize_X_column_wise(self,
                                  output_indicator = 'Trained_StandardScaler_X_ColWise',
                                  X_train=None,
                                  X_val=None,
                                  X_test=None,
                                  print_model=False):
        """
        Standardize feature variables (X) column-wise by removing the mean and scaling to unit variance.
        Transform the data such that each feature will have a mean value of 0 and a standard deviation of 1.

        Parameters
        -----------
        X_train : array
            Training feature matrix.
        X_val : array
            Validation feature matrix.
        X_test : array
            Test feature matrix.
        print_model : bool
            Whether to print the trained scaler model.

        Returns
        -------
        self.X_train_standardized_columnwise : array
            Standardized training feature matrix.
        self.X_val_standardized_columnwise : array
            Standardized validation feature matrix.
        self.X_test_standardized_columnwise : array
            Standardized test feature matrix.
        """

        # Set default values if None is provided
        X_train = self.X_train if X_train is None else X_train
        X_val = self.X_val if X_val is None else X_val
        X_test = self.X_test if X_test is None else X_test

        scaler_X = StandardScaler()

        self.X_train_standardized_columnwise = scaler_X.fit_transform(X_train)
        self.X_val_standardized_columnwise = scaler_X.transform(X_val)
        self.X_test_standardized_columnwise = scaler_X.transform(X_test)

        self.standardize_X_ColumnWise = scaler_X

        self.LoadSave.load_or_dump_trained_object(
            trained_object=self.standardize_X_ColumnWise,
            output_indicator=output_indicator,
            load_or_dump='dump')

        if print_model:
            print(scaler_X)

    def standardize_X_row_wise(self,
                               output_indicator = 'Trained_StandardScaler_X_RowWise',
                               X_train=None,
                               X_val=None,
                               X_test=None,
                               print_model=False):
        """
        Standardize feature variables (X) column-wise by removing the mean and scaling to unit variance.
        Transform the data such that each feature will have a mean value of 0 and a standard deviation of 1.

        Parameters
        ----------
        X_train : array
            Training feature matrix.
        X_val : array
            Validation feature matrix.
        X_test : array
            Test feature matrix.
        print_model : bool
            Whether to print the trained scaler model.

        Returns
        -------
        self.X_train_standardized_columnwise : array
            Standardized training feature matrix.
        self.X_val_standardized_columnwise : array
            Standardized validation feature matrix.
        self.X_test_standardized_columnwise : array
            Standardized test feature matrix.
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

        self.LoadSave.load_or_dump_trained_object(
            trained_object=self.standardize_X_RowWise,
            output_indicator=output_indicator,
            load_or_dump='dump')

        if print_model:
            print(scaler_X)

    def plot_boxplot_scaled_features(self, scaled_feature, title=None, xticks_list=None, fig_size=(14, 3)):
        """
        Make a boxplot with the scaled features.

        Description
        -----------
            - Median: middle quartile marks.
            - Inter-quartile range (The middle “box”): 50% of scores fall within the inter-quartile range.
            - Upper quartile: 75% of the scores fall below the upper quartile.
            - Lower quartile: 25% of scores fall below the lower quartile.
        """

        plt.figure(figsize=fig_size)
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
            xtick_positions = range(len(xticks_list))
            plt.xticks(xtick_positions, xticks_list)

        plt.tight_layout()
        plt.show()

    def plot_histogram_scaled_features(self, scaled_feature):
        """
        Plotting the histogram of scaled features
        """

        plt.figure(figsize=(15, 6))
        plt.hist(scaled_feature, bins=31)
        plt.xlabel('Scaled Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Scaled Features')
        plt.show()


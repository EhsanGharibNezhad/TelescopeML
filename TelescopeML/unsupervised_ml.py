# Import functions from other modules ============================
from IO_utils import LoadSave

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# Optimization libraries
from scipy import stats
import skopt
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective, plot_evaluations

# Data Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================================================================
# ==================                                           ==================
# ==================      Class TrainMlUnsupervised            ==================
# ==================                                           ==================
# ===============================================================================

class TrainMlUnsupervised:
    """
    Perform unsupervised learning tasks such as clustering, dimensionality reduction, and anomaly detection.

    Attributes
    -----------
    - feature_values: array
        Feature data for clustering or dimensionality reduction
    - feature_names: list
        Feature names
    - is_tuned: str
        To optimize the hyperparameters: 'yes' or 'no'
    - param_grid: dict
        Hyperparameters to be tuned
    - spectral_resolution: int
        Resolution of the synthetic spectra used to generate the dataset
    - is_feature_improved: str
        Options:
            'no': all features
            'pca': used Principal Component Analysis method for dimensionality reduction
    - n_jobs: int
        Number of processors for optimization step
    - cv: int
        Cross Validation
    - is_augmented: str
        Options:
            'no': used native dataset
            [METHOD]: augmented dataset like adding noise etc.
    - ml_model: object from sklearn package
        Use the library name to get instantiated e.g., KMeans()
    - ml_model_str: str
        Name of the ML model

    Outputs
    --------
    - Trained ML models and results
    """

    def __init__(self,
                 feature_values=None,
                 feature_names=None,
                 is_tuned='no',
                 param_grid=None,
                 spectral_resolution=100,
                 is_feature_improved='no',
                 n_jobs=-1,
                 cv=5,
                 is_augmented='no',
                 ml_model=None,
                 ml_model_str=''):
        if feature_values is None:
            feature_values = np.array([])  
        if feature_names is None:
            feature_names = []
        if param_grid is None:
            param_grid = {}  
        if ml_model is None:
            ml_model = KMeans() 

        self.feature_values = feature_values
        self.feature_names = feature_names
        self.is_tuned = is_tuned
        self.param_grid = param_grid
        self.spectral_resolution = spectral_resolution
        self.is_feature_improved = is_feature_improved
        self.n_jobs = n_jobs
        self.cv = cv
        self.is_augmented = is_augmented
        self.ml_model = ml_model
        self.ml_model_str = ml_model_str

    def load_and_preprocess_data(self, data=None, file_path=None):
        if file_path:
            df = pd.read_csv(file_path)
        elif data is not None:
            df = data
        else:
            raise ValueError("Either file_path or data must be provided.")
        
        df = df.select_dtypes(include=[np.number])
        return df
    
    def split_train_test(self, test_size=0.1):
        """
        Split the dataset into training and testing sets

        Inputs
        ------
            - self.feature_values
            - test_size: default is 10% kept for testing the final trained model
        Return
        -------
            - self.X_train, self.X_test: Used to train and evaluate the machine learning model
        """
        self.X_train, self.X_test = train_test_split(self.feature_values,
                                                    test_size=test_size,
                                                    shuffle=True,
                                                    random_state=42)

    def standardize_X_column_wise(self, X_train=None, X_test=None, print_model=False):
        """
        Standardize feature variables (X) column-wise by removing the mean and scaling to unit variance.

        Inputs:
            - X_train (numpy array): Training feature matrix
            - X_test (numpy array): Test feature matrix
            - print_model (bool): Whether to print the trained scaler model

        Assigns:
            - self.X_train_standardized_column_wise (numpy array): Standardized training feature matrix
            - self.X_test_standardized_column_wise (numpy array): Standardized test feature matrix
        """
        X_train = self.X_train if X_train is None else X_train
        X_test = self.X_test if X_test is None else X_test

        scaler_X = StandardScaler()
        self.X_train_standardized_column_wise = scaler_X.fit_transform(X_train)
        self.X_test_standardized_column_wise = scaler_X.transform(X_test)
        self.standardize_X_ColumnWise = scaler_X

        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned).load_or_dump_trained_object(
                     trained_object=self.standardize_X_ColumnWise,
                     output_indicator='standardize_X_ColumnWise',
                     load_or_dump='dump')

        if print_model:
            print(scaler_X)

    def normalize_X_column_wise(self, X_train=None, X_test=None):
        """
        Normalize feature variables (X) column-wise to the range [0, 1].

        Inputs:
            - X_train (numpy array): Training feature matrix
            - X_test (numpy array): Test feature matrix

        Assigns:
            - self.X_train_normalized_column_wise (numpy array): Normalized training feature matrix
            - self.X_test_normalized_column_wise (numpy array): Normalized test feature matrix
        """
        X_train = self.X_train if X_train is None else X_train
        X_test = self.X_test if X_test is None else X_test

        scaler_X = MinMaxScaler()
        self.X_train_normalized_column_wise = scaler_X.fit_transform(X_train)
        self.X_test_normalized_column_wise = scaler_X.transform(X_test)
        self.normalize_X_ColumnWise = scaler_X

        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned).load_or_dump_trained_object(
                     trained_object=self.normalize_X_ColumnWise,
                     output_indicator='normalize_X_ColumnWise',
                     load_or_dump='dump')

    def pca_transform(self, n_components=2):
        """
        Apply Principal Component Analysis (PCA) for dimensionality reduction

        Inputs:
            - n_components (int): Number of principal components to retain

        Assigns:
            - self.X_train_pca (numpy array): PCA-transformed training feature matrix
            - self.X_test_pca (numpy array): PCA-transformed test feature matrix
        """
        pca = PCA(n_components=n_components)
        self.X_train_pca = pca.fit_transform(self.X_train_standardized_column_wise)
        self.X_test_pca = pca.transform(self.X_test_standardized_column_wise)
        self.pca = pca

        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned).load_or_dump_trained_object(
                     trained_object=self.pca,
                     output_indicator='pca',
                     load_or_dump='dump')

    def dbscan_clustering(self, eps=0.5, min_samples=5):
        """
        Apply DBSCAN clustering algorithm

        Inputs:
            - eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other
            - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point

        Assigns:
            - self.dbscan_model (DBSCAN): Fitted DBSCAN model
            - self.dbscan_labels (numpy array): Cluster labels for each data point
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.dbscan_labels = dbscan.fit_predict(self.X_train_pca)
        self.dbscan_model = dbscan

        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned).load_or_dump_trained_object(
                     trained_object=self.dbscan_model,
                     output_indicator='dbscan_model',
                     load_or_dump='dump')

        print(f'DBSCAN Clustering Completed.')

    def plot_dbscan_clusters(self):
        """
        Plot the clusters formed by DBSCAN
        """
        if hasattr(self, 'X_train_pca') and hasattr(self, 'dbscan_labels'):
            plt.figure(figsize=(10, 6))
            plt.scatter(self.X_train_pca[:, 0], self.X_train_pca[:, 1], c=self.dbscan_labels, cmap='viridis')
            plt.title('DBSCAN Clustering')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.colorbar(label='Cluster Label')
            plt.show()
        else:
            print("DBSCAN model not fitted or PCA not performed.")

    def evaluate_dbscan_clusters(self):
        """
        Evaluate DBSCAN clustering performance
        """
        if hasattr(self, 'dbscan_labels'):
            num_clusters = len(set(self.dbscan_labels)) - (1 if -1 in self.dbscan_labels else 0)
            print(f'Number of clusters found: {num_clusters}')
            print(f'Number of noise points: {list(self.dbscan_labels).count(-1)}')
        else:
            print("DBSCAN model not fitted.")

    def kmeans_clustering(self, n_clusters=3):
        """
        Apply KMeans clustering algorithm

        Inputs:
            - n_clusters (int): The number of clusters to form

        Assigns:
            - self.kmeans_model (KMeans): Fitted KMeans model
            - self.kmeans_labels (numpy array): Cluster labels for each data point
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans_labels = kmeans.fit_predict(self.X_train_pca)
        self.kmeans_model = kmeans

        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned).load_or_dump_trained_object(
                     trained_object=self.kmeans_model,
                     output_indicator='kmeans_model',
                     load_or_dump='dump')

        print(f'KMeans Clustering Completed.')

    def plot_kmeans_clusters(self):
        """
        Plot the clusters formed by KMeans
        """
        if hasattr(self, 'X_train_pca') and hasattr(self, 'kmeans_labels'):
            plt.figure(figsize=(10, 6))
            plt.scatter(self.X_train_pca[:, 0], self.X_train_pca[:, 1], c=self.kmeans_labels, cmap='viridis')
            plt.title('KMeans Clustering')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.colorbar(label='Cluster Label')
            plt.show()
        else:
            print("KMeans model not fitted or PCA not performed.")

    def evaluate_kmeans_clusters(self):
        """
        Evaluate KMeans clustering performance
        """
        if hasattr(self, 'kmeans_labels'):
            silhouette_avg = silhouette_score(self.X_train_pca, self.kmeans_labels)
            print(f'Silhouette Score for KMeans clustering: {silhouette_avg}')
        else:
            print("KMeans model not fitted.")

    def fit_isolation_forest(self, contamination=0.1):
        """
        Fit Isolation Forest model for anomaly detection

        Inputs:
            - contamination (float): The proportion of outliers in the data set

        Assigns:
            - self.isolation_forest_model (IsolationForest): Fitted Isolation Forest model
            - self.isolation_forest_scores (numpy array): Anomaly scores for each data point
        """
        isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.isolation_forest_scores = isolation_forest.fit_predict(self.X_train_standardized_column_wise)
        self.isolation_forest_model = isolation_forest

        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned).load_or_dump_trained_object(
                     trained_object=self.isolation_forest_model,
                     output_indicator='isolation_forest_model',
                     load_or_dump='dump')

        print(f'Isolation Forest Model Fitted.')

    def plot_isolation_forest_scores(self):
        """
        Plot the anomaly scores given by Isolation Forest
        """
        if hasattr(self, 'isolation_forest_scores'):
            plt.figure(figsize=(10, 6))
            plt.hist(self.isolation_forest_scores, bins=50)
            plt.title('Isolation Forest Anomaly Scores')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
            plt.show()
        else:
            print("Isolation Forest model not fitted.")

    def fit_local_outlier_factor(self, n_neighbors=20):
        """
        Fit Local Outlier Factor model for anomaly detection

        Inputs:
            - n_neighbors (int): Number of neighbors to use by default for Local Outlier Factor

        Assigns:
            - self.local_outlier_factor_model (LocalOutlierFactor): Fitted Local Outlier Factor model
            - self.local_outlier_factor_scores (numpy array): Anomaly scores for each data point
        """
        local_outlier_factor = LocalOutlierFactor(n_neighbors=n_neighbors)
        self.local_outlier_factor_scores = local_outlier_factor.fit_predict(self.X_train_standardized_column_wise)
        self.local_outlier_factor_model = local_outlier_factor

        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned).load_or_dump_trained_object(
                     trained_object=self.local_outlier_factor_model,
                     output_indicator='local_outlier_factor_model',
                     load_or_dump='dump')

        print(f'Local Outlier Factor Model Fitted.')

    def plot_local_outlier_factor_scores(self):
        """
        Plot the anomaly scores given by Local Outlier Factor
        """
        if hasattr(self, 'local_outlier_factor_scores'):
            plt.figure(figsize=(10, 6))
            plt.hist(self.local_outlier_factor_scores, bins=50)
            plt.title('Local Outlier Factor Anomaly Scores')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
            plt.show()
        else:
            print("Local Outlier Factor model not fitted.")

    def tune_hyperparameters(self, search_space, n_iter=50):
        """
        Tune hyperparameters using Bayesian optimization

        Inputs:
            - search_space (dict): Dictionary defining the hyperparameter search space
            - n_iter (int): Number of iterations for optimization

        Assigns:
            - self.best_model (object): Best model with optimized hyperparameters
        """
        search = BayesSearchCV(self.ml_model, search_space, n_iter=n_iter, n_jobs=self.n_jobs, cv=self.cv)
        search.fit(self.X_train_standardized_column_wise)
        self.best_model = search.best_estimator_

        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned).load_or_dump_trained_object(
                     trained_object=self.best_model,
                     output_indicator='best_model',
                     load_or_dump='dump')

        print(f'Hyperparameters Tuned.')

    def plot_optimization_results(self, search_result):
        """
        Plot the results of the optimization process

        Inputs:
            - search_result (BayesSearchCV): The search result from Bayesian optimization
        """
        plot_evaluations(search_result)
        plt.show()

        plot_objective(search_result)
        plt.show()


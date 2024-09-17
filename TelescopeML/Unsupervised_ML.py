# Import functions from other modules ============================
# from io_funs import LoadSave

# Import python libraries ========================================

# ******* Standard Data Manipulation / Statistical Libraries *****
import numpy as np
import pandas as pd
from TelescopeML.DataMaster import DataProcessor
from typing import List, Union
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ******* Data Visualization Libraries ****************************
import matplotlib.pyplot as plt

# ******** Data science / Machine learning Libraries ***************

# Define the DBSCANProcessor class
class DBSCANProcessor(DataProcessor):
    """
    A class to perform DBSCAN clustering on the dataset.

    Inherits from DataProcessor to utilize existing data processing functionalities.

    Parameters
    ----------
    eps : float, optional (default=0.5)
        The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples : int, optional (default=5)
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
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
        ml_method: str = 'clustering',
        eps: float = 0.5,
        min_samples: int = 5
    ):
        super().__init__(
            flux_values,
            wavelength_names,
            wavelength_values,
            output_values,
            output_names,
            spectral_resolution,
            trained_ML_model,
            trained_ML_model_name,
            ml_method
        )
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan_model = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def fit(self, data=None):
        """
        Fit the DBSCAN model on the data.

        Parameters
        ----------
        data : array, optional
            The input data to fit the model. If None, it uses self.flux_values.

        Returns
        -------
        self : object
            Returns self.
        """
        if data is None:
            data = self.flux_values
        
        self.labels_ = self.dbscan_model.fit_predict(data)
        return self

    def get_core_samples(self):
        """
        Get the indices of the core samples.

        Returns
        -------
        core_sample_indices_ : array, shape (n_core_samples,)
            Indices of core samples.
        """
        return self.dbscan_model.core_sample_indices_

    def get_labels(self):
        """
        Get the labels for each point in the dataset.

        Returns
        -------
        labels_ : array, shape (n_samples,)
            Cluster labels for each point in the dataset. Noisy samples are given the label -1.
        """
        return self.labels_

    def plot_clusters(self):
        """
        Visualize the DBSCAN clusters using a scatter plot.
        """
        plt.figure(figsize=(10, 6))
        unique_labels = set(self.labels_)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (self.labels_ == k)
            xy = self.flux_values[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

        plt.title('DBSCAN Clustering')
        plt.show()


class KMeansClustering:
    """
    A class to perform KMeans clustering on a given dataset.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.
    random_state : int, optional
        Determines random number generation for centroid initialization.
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_model = None
        self.labels = None
        self.cluster_centers = None
        self.inertia = None
        self.silhouette_score_ = None

    def fit(self, data: np.ndarray):
        """
        Fit the KMeans model to the data.

        Parameters
        ----------
        data : np.ndarray
            The input data for clustering.
        """
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.labels = self.kmeans_model.fit_predict(data)
        self.cluster_centers = self.kmeans_model.cluster_centers_
        self.inertia = self.kmeans_model.inertia_
        self.silhouette_score_ = silhouette_score(data, self.labels)

    def plot_clusters(self, data: np.ndarray, feature_indices: List[int] = [0, 1]):
        """
        Plot the clusters with the selected features.

        Parameters
        ----------
        data : np.ndarray
            The input data for clustering.
        feature_indices : list of int, optional
            Indices of the features to plot (default is [0, 1]).
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, feature_indices[0]], data[:, feature_indices[1]], c=self.labels, cmap='viridis', marker='o', edgecolor='k')
        plt.scatter(self.cluster_centers[:, feature_indices[0]], self.cluster_centers[:, feature_indices[1]], s=300, c='red', marker='X')
        plt.xlabel(f'Feature {feature_indices[0]}')
        plt.ylabel(f'Feature {feature_indices[1]}')
        plt.title('KMeans Clustering')
        plt.show()

    def plot_elbow_method(self, data: np.ndarray, max_clusters: int = 10):
        """
        Plot the elbow method to determine the optimal number of clusters.

        Parameters
        ----------
        data : np.ndarray
            The input data for clustering.
        max_clusters : int, optional
            Maximum number of clusters to test (default is 10).
        """
        inertias = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_clusters + 1), inertias, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal Number of Clusters')
        plt.show()

    def get_cluster_info(self):
        """
        Get information about the clusters.

        Returns
        -------
        dict
            A dictionary containing the labels, cluster centers, inertia, and silhouette score.
        """
        return {
            'labels': self.labels,
            'cluster_centers': self.cluster_centers,
            'inertia': self.inertia,
            'silhouette_score': self.silhouette_score_
        }


class PCAProcessor:
    """
    Perform Principal Component Analysis (PCA) on the dataset.

    This class provides methods to apply PCA on the data, transform it into the reduced feature space,
    and visualize the explained variance by the principal components.

    Parameters
    ----------
    n_components : int, optional
        Number of principal components to keep (default is None, meaning all components are kept).

    Attributes
    ----------
    pca : PCA object
        The fitted PCA model.
    explained_variance_ratio_ : array
        Percentage of variance explained by each of the selected components.
    components_ : array
        Principal axes in feature space.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca = None
        self.explained_variance_ratio_ = None
        self.components_ = None

    def fit(self, X):
        """
        Fit the PCA model to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to fit PCA on.
        """
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.components_ = self.pca.components_

    def transform(self, X):
        """
        Apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_components)
            The transformed data.
        """
        if self.pca is None:
            raise RuntimeError("You need to fit the PCA model before transforming data.")
        return self.pca.transform(X)

    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to fit and transform.

        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_components)
            The transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def plot_explained_variance(self):
        """
        Plot the explained variance ratio of each principal component.
        """
        if self.explained_variance_ratio_ is None:
            raise RuntimeError("You need to fit the PCA model before plotting explained variance.")
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(self.explained_variance_ratio_) + 1), self.explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Principal Components')
        plt.show()

    def plot_principal_components(self, X_transformed, y=None):
        """
        Plot the data in the space of the first two principal components.

        Parameters
        ----------
        X_transformed : array-like, shape (n_samples, n_components)
            The data transformed into the PCA space.
        y : array-like, shape (n_samples,), optional
            Labels for coloring the data points (default is None, meaning no coloring).
        """
        if X_transformed.shape[1] < 2:
            raise RuntimeError("Need at least 2 principal components for this plot.")
        
        plt.figure(figsize=(10, 6))
        if y is None:
            plt.scatter(X_transformed[:, 0], X_transformed[:, 1])
        else:
            plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap='viridis', alpha=0.7)
            plt.colorbar()
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Data in the Space of the First Two Principal Components')
        plt.show()

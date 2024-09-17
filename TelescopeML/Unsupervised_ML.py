# unsupervised_ml.py

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np

import matplotlib.pyplot as plt

def perform_pca(data, n_components):
    """
    Perform Principal Component Analysis (PCA) on the given data.

    Parameters:
    - data (numpy.ndarray or pandas.DataFrame): The input data to perform PCA on.
    - n_components (int): The number of principal components to compute.

    Returns:
    - pca (PCA object): The fitted PCA object.
    - components (numpy.ndarray): The transformed data in the principal component space.
    """
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data)
    return pca, components

def perform_clustering(components, algorithm, params):
    """
    Perform clustering on the PCA-transformed data using the specified algorithm.

    Parameters:
    - components (numpy.ndarray): The PCA-transformed data.
    - algorithm (str): The clustering algorithm to use ('K-Means', 'DBSCAN', 'Gaussian Mixture').
    - params (dict): A dictionary of parameters for the chosen clustering algorithm.

    Returns:
    - cluster_labels (numpy.ndarray or list): The labels assigned to each data point.
    - cluster_centers (numpy.ndarray or None): The coordinates of cluster centers, if applicable.
    """
    if algorithm == 'K-Means':
        model = KMeans(n_clusters=params.get('n_clusters', 3), random_state=42)
    elif algorithm == 'DBSCAN':
        model = DBSCAN(eps=params.get('eps', 0.5), min_samples=params.get('min_samples', 5))
    elif algorithm == 'Gaussian Mixture':
        model = GaussianMixture(n_components=params.get('n_components', 3), random_state=42)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Fit the model and predict cluster labels
    cluster_labels = model.fit_predict(components)

    # Extract cluster centers if applicable
    if algorithm == 'K-Means':
        cluster_centers = model.cluster_centers_
    elif algorithm == 'Gaussian Mixture':
        cluster_centers = model.means_
    else:
        cluster_centers = None  # DBSCAN does not have cluster centers

    return cluster_labels, cluster_centers

def plot_clusters(components, cluster_labels, cluster_centers=None):
    """
    Plot the clustering results for the first two principal components.

    Parameters:
    - components (numpy.ndarray): The PCA-transformed data.
    - cluster_labels (numpy.ndarray or list): The labels assigned to each data point.
    - cluster_centers (numpy.ndarray or None): The coordinates of cluster centers, if applicable.
    """
    plt.figure(figsize=(8, 6))
    
    # Plot the points with their cluster labels
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        cluster_points = components[cluster_labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', s=50)
    
    # Plot cluster centers if available
    if cluster_centers is not None:
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Centers')
    
    plt.title('Clustered Data in First Two PCA Components')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()
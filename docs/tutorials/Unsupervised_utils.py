import os
import pickle as pk
import tensorflow as tf
from typing import List, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


class Unsupervised_Data_Processor:
    def __init__(self,
            flux_values: Union[np.ndarray] = None,
            wavelength_names: Union[List[str]] = None,
            wavelength_values: Union[np.ndarray] = None,
            output_values: Union[np.ndarray] = None,
            output_names: Union[str] = None,
            spectral_resolution: Union[None, int] = None):

        self.flux_values = flux_values
        self.wavelength_names = wavelength_names
        self.wavelength_values = wavelength_values
        self.output_values = output_values
        self.output_names = output_names
        self.spectral_resolution = spectral_resolution 
        
    def normalize_X_column_wise(self, X=None):
        X = self.flux_values if X is None else X
        
        normalizer = MinMaxScaler(feature_range=(0, 1))
        self.X_normalized_columnwise = normalizer.fit_transform(X)
        
        return self.X_normalized_columnwise
        
    def normalize_X_row_wise(self, X=None):
        X = self.flux_values if X is None else X
        
        normalizer = MinMaxScaler(feature_range=(0, 1))
        self.X_normalized_rowwise = normalizer.fit_transform(X.T).T
        
        return self.X_normalized_rowwise
        
    def standardize_X_column_wise(self, X=None):
        X = self.flux_values if X is None else X
        
        scaler_X = StandardScaler()
        self.X_standardized_column_wise = scaler_X.fit_transform(X)
        
        return self.X_standardized_column_wise
    
    def standardize_X_row_wise(self, X=None):
        X = self.flux_values if X is None else X
        
        scaler_X = StandardScaler()
        self.X_standardized_row_wise = scaler_X.fit_transform(X.T).T
        
        return self.X_standardized_row_wise
        
           
class Unsupervised_Algorithms():
    def __init__(self, X):
        self.X = X
    
    def kmeans(self, num_clusters, max_iter, f1, f2):
        km = KMeans(n_clusters=4, max_iter = 1000, n_init='auto')
        km.fit(self.X)
        y_kmeans = km.predict(self.X)
        fig, ax = plt.subplots(1,1)
        ax.scatter(self.X[:, f1], self.X[:, f2], c=y_kmeans, s=30, cmap='viridis')
        centers = km.cluster_centers_
        ax.scatter(centers[:, f1], centers[:, f2], c='black', s=200, alpha=0.5);
        ax.set_title(f'K-Means Clustering for features {f1} and {f2}')
        return fig
    
    def pca(self, num_components):
        pca = PCA(n_components=num_components)
        Xt = pca.fit_transform(self.X)
        fig, ax = plt.subplots(1,1)
        ax.scatter(Xt[:,0], Xt[:,1])
        ax.set_title('PCA')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        return fig

   
    def dbscan(self, eps, min_sample, f1, f2):
        data_2_features = self.X[:, [f1, f2]]

        dbscan = DBSCAN(eps=eps, min_samples=min_sample)
        dbscan_labels = dbscan.fit_predict(data_2_features)

        fig, ax = plt.subplots(1,1, figsize=(5, 4))

        unique_labels = set(dbscan_labels)
        for label in unique_labels:
            if label == -1:  # Noise points
                color = 'slategrey'  
                label_name = 'Noise'
                ax.scatter(data_2_features[dbscan_labels == label, 0], 
                            data_2_features[dbscan_labels == label, 1], 
                            c=color, label=label_name, s=15, alpha=0.01)
            else:  # Cluster points
                color = plt.cm.nipy_spectral(float(label) / len(unique_labels))
                label_name = f'Cluster {label}'
                plt.scatter(data_2_features[dbscan_labels == label, 0], 
                            data_2_features[dbscan_labels == label, 1], 
                            c=[color], label=label_name, s=50, alpha=1.0)  

        ax.set_title('DBSCAN Clustering with Two Features')
        plt.legend()
        ax.set_xlabel(f'Feature {f1}')
        ax.set_ylabel(f'Feature {f2}')
        return fig
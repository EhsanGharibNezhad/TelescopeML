#Collection of unsupervised learning techniques for TelescopeML
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import spectral_clustering
import os
__reference_data__ = os.getenv("TelescopeML_reference_data")
__reference_data_path__ = __reference_data__




class UnsupervisedML:
    """
    Provide unsupervised learning techniques that can be used for TelescopeML.
    The algorithms provided are as follows:
        Dimensionality Reduction:
            - Principal Component Analysis (PCA)
            - t-distributed stochastic neighbor embedding (t-SNE)
        Clustering:
            - K-Means Clustering
            - Density-based spatial clustering of applications with noise (DBSCAN)
            - Spectral Clustering
    
    Parameters
    ----------
    dataset : pd.DataFrame
        Dataframe to be used to train an unsupervised model on.
    
    """
    def __init__(self, dataset):
        self.dataset = dataset
    
    
    def pca(self, train_data=None, val_data=None, test_data=None, n_components=10, random_state=42):
        """
        Generates a dataframe of principal components.
    
        Parameters
        ----------
        train_data : pd.DataFrame
            Dataset to fit the PCA model to.
        val_data : pd.DataFrame
            (optional) Validation dataset to apply the PCA transform on.
        test_data : pd.DataFrame
            (optional) Testing dataset to apply the PCA transform on.
        n_components : int
            Number of principal components to generate and return.
    
        """
        pca_model = PCA(n_components = n_components,
                        random_state = random_state)
        self.pca_train_dataset = pd.DataFrame(pca_model.fit_transform(train_data))
        self.pca_explained_variance_ratio = pca_model.explained_variance_ratio_
        self.total_varaince_captured = np.sum(self.pca_explained_variance_ratio)
        if val_data != None:
            self.pca_val_dataset = pd.DataFrame(pca_model.transform(val_data))
        if test_data != None:
            self.pca_test_dataset = pd.DataFrame(pca_model.transform(test_data))
        return
    
    
    def tsne(self, train_data=None, val_data=None, test_data=None, n_components = 3, learning_rate='auto', perplexity=100, random_state=42):
        """
        Generates a dataframe of t-SNE embeddings.
    
        Parameters
        ----------
        train_data : pd.DataFrame
            Dataset to generate the t-SNE embeddings from.
        val_data : pd.DataFrame
            (optional) Validation dataset to apply the t-SNE embeddings on.
        test_data : pd.DataFrame
            (optional) Testing dataset to apply the t-SNE embeddings on.
        n_components : int
            The dimension size of the embedded space.
        learning_rate : float
            The learning rate.
        perplexity : int
            The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms.
            Larger datasets usually require a larger perplexity. Consider choosing a value between 5 and 200.
    
        """
        tsne_model = TSNE(n_components = n_components,
                          learning_rate = learning_rate,
                          perplexity = perplexity,
                          random_state = random_state)
        self.tsne_train_dataset = pd.DataFrame(tsne_model.fit_transform(train_data))
        if val_data != None:
            self.tsne_val_dataset = pd.DataFrame(tsne_model.transform(val_data))
        if test_data != None:
            self.tsne_test_dataset = pd.DataFrame(tsne_model.transform(test_data))
        return
        
    
    def kmeans(self, train_data=None, val_data=None, test_data=None, n_clusters = 8, n_init = 50, max_iter=500, random_state=42):
        """
        Perform K-means clustering on a dataset. Typically, clustering will be performed after a dimensionality reduction technique has been applied.
    
        Parameters
        ----------
        train_data : pd.DataFrame
            Dataset to train the clustering model on.
        val_data : pd.DataFrame
            (optional) Validation dataset to apply the clusterings to.
        test_data : pd.DataFrame
            (optional) Testing dataset to apply the clusterings to.
        n_clusters : int
            The number of clusters to generate.
        n_init : int
            The number of times the algorithm is randonly instantiated (this will help ensure the best clustering is found).
        max_iter : int
            The maximum number of iterations to train for.
    
        """
        self.kmeans_clusters = n_clusters
        kmeans_model = KMeans(n_clusters = n_clusters,
                              n_init = n_init,
                              max_iter = max_iter,
                              random_state = random_state)
        kmeans_model.fit(train_data)
        #Add the clusterings to the datasets
        self.train_kmeans_clustered = train_data.copy()
        self.train_kmeans_clustered['cluster'] = kmeans_model.fit_predict(train_data)
        self.distortion = kmeans_model.inertia_
        if val_data != None:
            self.val_kmeans_clustered = val_data.copy()
            self.val_kmeans_clustered['cluster'] = kmeans_model.fit_predict(val_data)
        if test_data != None:
            self.test_kmeans_clustered = test_data.copy()
            self.test_kmeans_clustered = kmeans_model.fit_predict(test_data)
        return
        
    
    def kmeans_elbow_plot(self, train_data=None, max_k = 20, n_init = 50, max_iter=500, random_state=42):
        """
        Utilize the 'elbow' method to identify the trade-off curve between the number of clusters and the distortion for a K-Means model.
        Generate an elbow plot for a number of max_k clusters.
    
        Parameters
        ----------
        train_data : pd.DataFrame
            Dataset to train the clustering model on.
        max_k : int
            The total number of clusters sizes to test.
        n_init : int
            The number of times the algorithm is randonly instantiated (this will help ensure the best clustering is found).
        max_iter : int
            The maximum number of iterations to train for.
        
        """
        distortion_list = []
        k_list = []
        #Compute the distortions for all clusters
        for k in range(max_k):
            k_list.append(k+1)
            n_clusters = k+1
            kmeans_model = KMeans(n_clusters = n_clusters,
                                  n_init = n_init,
                                  max_iter = max_iter,
                                  random_state = random_state)
            kmeans_model.fit(train_data)
            distortion = kmeans_model.inertia_
            distortion_list.append(distortion)
        
        #Generate the elbow plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=k_list,y=distortion_list)
        sns.scatterplot(x=k_list,y=distortion_list)
        ax.set_title("K-Means Clustering Elbow Plot")
        ax.set_xlabel('K Clusters')
        ax.set_ylabel('Distortion')
        return
    
    
    def dbscan(self, train_data=None, val_data=None, test_data=None, eps=0.5, min_samples=10, algorithm='brute', p=2):
        """
        Generate DBSCAN clusters on a dataset. Typically, clustering will be performed after a dimensionality reduction technique has been applied.
    
        Parameters
        ----------
        train_data : pd.DataFrame
            Dataset to train the clustering model on.
        val_data : pd.DataFrame
            (optional) Validation dataset to apply the clusterings to.
        test_data : pd.DataFrame
            (optional) Testing dataset to apply the clusterings to.
        eps : float
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples : int
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            If set to a higher value, DBSCAN will find denser clusters, whereas if it is set to a lower value, the found clusters will be more sparse.
        algorithm : str
            The algorithmto be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors
            ('auto', 'ball_tree', 'kd_tree', 'brute').
        p : int
            The power of the Minkowski metric to be used to calculate distance between points. If None, then p=2 (equivalent to the Euclidean distance).
    
        """
        dbscan_model = DBSCAN(eps = eps,
                              min_samples = min_samples,
                              algorithm = algorithm,
                              p = 2)
        dbscan_model.fit(train_data)
        #Add the clusterings to the datasets
        self.train_dbscan_clustered = train_data.copy()
        self.train_dbscan_clustered['cluster'] = dbscan_model.fit_predict(train_data)
        if val_data != None:
            self.val_dbscan_clustered = val_data.copy()
            self.val_dbscan_clustered['cluster'] = dbscan_model.fit_predict(val_data)
        if test_data != None:
            self.test_dbscan_clustered = test_data.copy()
            self.test_dbscan_clustered = dbscan_model.fit_predict(test_data)
        return
    
    
    def spec_c(self, train_data=None, affinity_metric='rbf', n_clusters=8, assign_labels="discretize", random_state=42):
        """
        Generate spectral clusters on a dataset. Typically, clustering will be performed after a dimensionality reduction technique has been applied.
    
        Parameters
        ----------
        train_data : pd.DataFrame
            Dataset to train the clustering model on.
        affinity_metric : str
            The metric to be used to compute pairwise distances between points in the data
            (‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’, ‘laplacian’, ‘sigmoid’, ‘cosine’)
            (reference https://scikit-learn.org/stable/modules/metrics.html#metrics for the metrics).
        n_clusters : int
            The number of clusters to generate.
        assign_labels : str
            The strategy to use to assign labels in the embedding space ('kmeans', 'discretize', 'cluster_qr').
    
        """
        self.n_spec_clusters = n_clusters
        #Measures of similarity between points in the dataset
        affinity = pairwise_kernels(train_data,
                                    metric=affinity_metric) 
        #Compute the spectral clusters
        spec_clusters = spectral_clustering(affinity=affinity,
                                         n_clusters=8,
                                         assign_labels="discretize",
                                         random_state=42)
        self.train_spec_c_clustered = train_data.copy()
        self.train_spec_c_clustered['cluster'] = spec_clusters
        return
        
    
    def scatter_3d_cluster_plot(self, dataset=None, method=None):
        """
        Generate a 3d scatterplot with clusters. Should be run after one has obtained a clustering.
    
        Parameters
        ----------
        dataset : pd.DataFrame
            Dataset with first 3 features that will be plotted
        method : str
            This will define the type of title and axis labels to generate depending on the dimensionality technique and clustering used.
    
        """
        #Generate a 3-D plot visualizing the top 3 PCs
        fig = plt.figure() #plt.figure(figsize=(10, 7))
        ax = plt.subplot(111, projection='3d')
        # Scatter plot with color based on the cluster labels
        scatter = ax.scatter(dataset[0], dataset[1], dataset[2], c=dataset['cluster'], cmap='inferno')
        fig.colorbar(scatter).set_label('Clusters')
        
        # Adding titles and labels
        if method == 'PCA_KMeans':
            ax.set_title(f"3D K-Means Clustering of PCA Spectra Vectors for $K={self.kmeans_clusters}$ Clusters")
            ax.set_xlabel('PC-1')
            ax.set_ylabel('PC-2')
            ax.set_zlabel('PC-3')
        elif method == 'PCA_DBSCAN':
            ax.set_title("3D DBSCAN Clustering of PCA Spectra Vectors")
            ax.set_xlabel('PC-1')
            ax.set_ylabel('PC-2')
            ax.set_zlabel('PC-3')
        elif method == 'PCA_Spec_c':
            ax.set_title(f"3D Spectral Clustering of PCA Spectra Vectors for ${self.n_spec_clusters}$ Clusters")
            ax.set_xlabel('PC-1')
            ax.set_ylabel('PC-2')
            ax.set_zlabel('PC-3')
        elif method == 'TSNE_KMeans':
            ax.set_title(f"3D K-Means Clustering of t-SNE Spectra Embeddings for $K={self.kmeans_clusters}$ Clusters")
            ax.set_xlabel('tSNE-1')
            ax.set_ylabel('tSNE-2')
            ax.set_zlabel('tSNE-3')
        elif method == 'TSNE_DBSCAN':
            ax.set_title("3D DBSCAN Clustering of t-SNE Spectra Embeddings")
            ax.set_xlabel('tSNE-1')
            ax.set_ylabel('tSNE-2')
            ax.set_zlabel('tSNE-3')
        elif method == 'TSNE_Spec_c':
            ax.set_title(f"3D Spectral Clustering of t-SNE Spectra Embeddings for ${self.n_spec_clusters}$ Clusters")
            ax.set_xlabel('tSNE-1')
            ax.set_ylabel('tSNE-2')
            ax.set_zlabel('tSNE-3')
        return fig




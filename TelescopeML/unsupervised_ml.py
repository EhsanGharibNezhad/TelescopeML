import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class UnsupervisedMethods:
    """"
    Different methods for performing unsupervised learning.

    """
    def __init__(self, dataframe):
        """
        input:
            dataframe

        returns: 
            None
        """
        self.data = dataframe

    def kmeans_clustering(self, n_clusters=3):
        """
        Perform k-means clustering

        input:
            n-clusters: int, the number of clusters to identify in the deta (default 3).

        Returns:
            Modified dataframe with "cluster" column added that assign each data point to a cluster
        """
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        self.data['cluster'] = kmeans.fit_predict(self.data)
        return self.data

    def dbscan_clustering(self, eps=0.5, min_samples=5):
        """
        Performs DBCAN clustering

        inputs:
            eps(float): the maximum distance between two points to be considered neighbors(default:0.5)
            min_sampled(int): the minimum number of neighbors a point needs to a considered a core point(default:5)

        Returns:
            Modified dataframe with "cluster" column added that assign each data point to a cluster
        
        """

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.data['cluster'] = dbscan.fit_predict(self.data)
        return self.data

    def pca_analysis(self, n_components=2):

        """
        Perform PCA analysis 

        inputs:
            n_components(int): number of principle components to retain during

        returns:
            new dataframe contaning the data projected onto the number of principle components
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
        return pca_df

    def visualize_clustering(self, method='kmeans', use_pca=False, x_col='gravity', y_col='temperature', n_clusters=3, eps=0.5, min_samples=5):
        """
        Visualize the different clustering methods using matplotlib

        inputs:
            method(str): the clustering method to use(default: 'kmeans')
            use_pca(bool): wether or to perform PCA before clustering or not(default: False)
            x_col(str): the name of the feature/column to use for the x-axis(default:'gravity')
            y_col(str): the name of the feature/column to use for the y-axis(default:'temperature')
            eps(float): the epsilon parameter for DBCAN (default: 0.5), only applies then method is dbscan
            min_samples(int)the minimum number of samples parameter for DBSCAN (default: 5) only applies then method is dbscan

        returns: 
            matplotlib figure object
        """

        if use_pca:
            pca_df = self.pca_analysis(n_components=2)
            self.data = pca_df

        if method == 'kmeans':
            self.kmeans_clustering(n_clusters=n_clusters)
        elif method == 'dbscan':
            self.dbscan_clustering(eps=eps, min_samples=min_samples)

        if use_pca:
            x_col, y_col = 'PC1', 'PC2'

        fig, ax = plt.subplots(figsize=(10, 6))
        for cluster in self.data['cluster'].unique():
            cluster_data = self.data[self.data['cluster'] == cluster]
            ax.scatter(cluster_data[x_col], cluster_data[y_col], label=f'Cluster {cluster}')

        ax.set_title(f'{method.upper()} Clustering Visualization')
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel(y_col.capitalize())
        return fig
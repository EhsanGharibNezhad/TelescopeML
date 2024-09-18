from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class UnsupervisedML:
    def __init__(self):
        pass

    def kmeans_clustering(self, data, n_clusters, init="k-means++"):
        """
        Perform K-Means clustering on the dataset.
        n_clusters: Number of clusters to form.
        init: Method for initialization.

        Returns the KMeans object.
        """
        kmeans = KMeans(n_clusters=n_clusters, init=init, random_state=42)
        kmeans.fit(data)
        return kmeans

    def dbscan_clustering(self, data, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering.
        eps: The maximum distance between two samples for them to be considered in the same neighborhood.
        min_samples: The number of samples in a neighborhood for a point to be considered a core point.

        Returns the DBSCAN object.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(data)
        return dbscan

    def reduce_dataset_pca(self, data, n_components=2):
        """
        Perform PCA on the dataset to reduce dimensionality.
        n_components: Number of principal components to keep.

        Returns the transformed data.
        """
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(data)
        return X_pca

    def reduce_dataset_tsne(self, data, n_components=2):
        """
        Perform t-SNE on the dataset to reduce dimensionality.
        n_components: Number of components to keep.

        Returns the transformed data.
        """
        tsne = TSNE(n_components=n_components)
        X_tsne = tsne.fit_transform(data)
        return X_tsne

    def plot_kmeans_clusters(self, data, labels, centers):
        fig = plt.figure(figsize=(10, 8))

        # Check if the data is 2D or 3D
        if data.shape[1] == 3:
            ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
            ax.set_title("KMeans Clustering (3D)")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")

            ax.scatter(
                data[:, 0], data[:, 1], data[:, 2], c=labels, cmap="viridis", s=25
            )
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                centers[:, 2],
                marker="X",
                s=100,
                color="black",
            )
        elif data.shape[1] == 2:
            ax = fig.add_subplot(111)
            ax.set_title("KMeans Clustering (2D)")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

            ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis", s=25)
            ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=100, color="black")

        return fig

    def plot_dbscan_clusters(self, data, labels, core_samples_mask, noise_mask):
        fig = plt.figure(figsize=(10, 8))

        if data.shape[1] == 3:
            ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
            ax.set_title("DBSCAN Clustering (3D)")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")

            ax.scatter(
                data[core_samples_mask, 0],
                data[core_samples_mask, 1],
                data[core_samples_mask, 2],
                c=labels[core_samples_mask],
                cmap="viridis",
                s=25,
            )
            ax.scatter(
                data[noise_mask, 0],
                data[noise_mask, 1],
                data[noise_mask, 2],
                c="red",
                s=25,
                label="Noise Points",
            )
        elif data.shape[1] == 2:
            ax = fig.add_subplot(111)
            ax.set_title("DBSCAN Clustering (2D)")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

            ax.scatter(
                data[core_samples_mask, 0],
                data[core_samples_mask, 1],
                c=labels[core_samples_mask],
                cmap="viridis",
                s=25,
            )
            ax.scatter(
                data[noise_mask, 0],
                data[noise_mask, 1],
                c="red",
                s=25,
                label="Noise Points",
            )

        plt.legend()
        return fig

    def plot_dimension_reduction(
        self, data, method, output_labels=None, output_var=None
    ):
        """
        Plot the reduced dataset using PCA or t-SNE. Overlay output labels if provided.
        """
        plt.figure(figsize=(10, 8))

        if data.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

            if output_labels is None:
                scatter = ax.scatter(
                    data[:, 0], data[:, 1], data[:, 2], cmap="viridis", s=25
                )
            else:
                scatter = ax.scatter(
                    data[:, 0],
                    data[:, 1],
                    data[:, 2],
                    c=output_labels,
                    cmap="viridis",
                    s=25,
                )
                fig.colorbar(scatter, ax=ax, label=output_var)

            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            ax.set_title(f"3D Data Visualization: {method} Analysis")

        elif data.shape[1] == 2:
            fig, ax = plt.subplots()

            if output_labels is None:
                scatter = ax.scatter(data[:, 0], data[:, 1], cmap="viridis", s=25)
            else:
                scatter = ax.scatter(
                    data[:, 0], data[:, 1], c=output_labels, cmap="viridis", s=25
                )
                fig.colorbar(scatter, ax=ax, label=output_var)

            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_title(f"2D Data Visualization: {method} Analysis")

        return fig

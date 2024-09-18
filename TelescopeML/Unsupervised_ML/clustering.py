#class for two clustering methods, DB Scan and K-means
from sklearn.cluster import DBSCAN, KMeans

class clustering:
    def apply_dbscan(self, data, eps=0.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(data)
        return dbscan.labels_
    
    def apply_kmeans(self, data, n_clusters=3, random_state=42):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans.fit(data)
        return kmeans.labels_
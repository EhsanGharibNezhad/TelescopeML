#This file is for two methods of dimensionality reduction: PCA and t-SNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

class dr:
    def pca(self, data, n_components, scale_temp=None, scale_metallicity=None, scale_gravity=None, scale_c_o_ratio=None):
        if scale_temp: data['temperature'] = np.log10(data['temperature'])
        if scale_metallicity: data['metallicity'] = np.log10(data['metallicity'])
        if scale_gravity: data['gravity'] = np.log10(data['gravity'])
        if scale_c_o_ratio: data['c_o_ratio'] = np.log10(data['c_o_ratio'])

        # Initialize the PCA model
        pca = PCA(n_components=n_components)

        # Fit and transform the data
        pca_data = pca.fit_transform(data)
        
        return pca_data, pca.explained_variance_ratio_

    def tsne(self, data, n_components=2, scale_temp=None, scale_metallicity=None, scale_gravity=None, scale_c_o_ratio=None, perplexity=30, n_iter=300, random_state=42):
        if scale_temp: data['temperature'] = np.log10(data['temperature'])
        if scale_metallicity: data['metallicity'] = np.log10(data['metallicity'])
        if scale_gravity: data['gravity'] = np.log10(data['gravity'])
        if scale_c_o_ratio: data['c_o_ratio'] = np.log10(data['c_o_ratio'])

        # Initialize the t-SNE model
        tsne_model = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    
        # Fit and transform the data
        tsne_data = tsne_model.fit_transform(data)
    
        return tsne_data
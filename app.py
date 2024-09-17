import streamlit as st
import pandas as pd
from TelescopeML.unsupervised_ml import *
import matplotlib.pyplot as plt
import os 

os.environ["TelescopeML_reference_data"] = "C:\\Users\\husse\\Desktop\\reference_data"

__reference_data_path__ = os.getenv("TelescopeML_reference_data")
__reference_data_path__ 


# Note: insert the directory of the reference_data if you get an error reading the reference data!!!
# __reference_data_path__ = 'INSERT_DIRECTORY_OF_reference_data'


data = pd.read_csv(os.path.join(__reference_data_path__, 
                                    'training_datasets', 
                                    'browndwarf_R100_v4_newWL_v3.csv.bz2'), compression='bz2')

st.title('TelescopeML Unsupervised Learning')
unsupervised = UnsupervisedMethods(data)
method = st.selectbox('Choose Method', ['K-means', 'DBSCAN'])
use_pca = st.checkbox('Use PCA')

if method == 'K-means':
    unsupervised = UnsupervisedMethods(data)
    n_clusters = st.slider('Number of clusters', 2, 10, 2)
    if use_pca:
        x_col, y_col = 'PC1', 'PC2'
    if not use_pca:
        x_col = st.selectbox('Choose x-axis column', ['gravity', 'temperature', 'c_o_ratio', 'metallicity'])
        y_col = st.selectbox('Choose y-axis column', ['gravity', 'temperature', 'c_o_ratio', 'metallicity'])
    if st.button('Show results'):
        fig, ax = plt.subplots()
        fig = unsupervised.visualize_clustering(method='kmeans', use_pca=use_pca, x_col=x_col, y_col=y_col, n_clusters=n_clusters)
        st.pyplot(fig)

elif method == 'DBSCAN':
    unsupervised = UnsupervisedMethods(data)
    eps = st.slider('Epsilon', 0.1, 1.0, 0.5)
    min_samples = st.slider('Minimum samples', 1, 5, 10)
    if use_pca:
        x_col, y_col = 'PC1', 'PC2'
    if not use_pca:
        x_col = st.selectbox('Choose x-axis column', ['gravity', 'temperature', 'c_o_ratio', 'metallicity'])
        y_col = st.selectbox('Choose y-axis column', ['gravity', 'temperature', 'c_o_ratio', 'metallicity'])
    if st.button('Show results'):
        fig, ax = plt.subplots()
        fig = unsupervised.visualize_clustering(method='dbscan', use_pca=use_pca, x_col=x_col, y_col=y_col, eps=eps, min_samples=min_samples)
        st.pyplot(fig)




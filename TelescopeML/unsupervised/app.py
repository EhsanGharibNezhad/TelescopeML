import numpy as np
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from unsupervised_ml import UnsupervisedML

from sklearn.preprocessing import StandardScaler


@st.cache_data
def import_data():
    """
    Import the Brown Dwarf dataset and perform data preprocessing.
    TODO: Allow user to upload their own dataset.

    Returns the preprocessed dataset and output variables.
    """
    __reference_data_path__ = os.getenv("TelescopeML_reference_data")
    train_BD = pd.read_csv(
        os.path.join(
            __reference_data_path__,
            "training_datasets",
            "browndwarf_R100_v4_newWL_v3.csv.bz2",
        ),
        compression="bz2",
    )

    train_BD = train_BD.sample(frac=0.5, random_state=42)

    output_names = ["gravity", "temperature", "c_o_ratio", "metallicity"]
    X = train_BD.drop(columns=output_names)
    y = train_BD[output_names]
    # log-transform the 'temperature' variable toreduce the skewness of the data, making it more symmetric and normal-like for the ML model
    y.loc[:, "temperature"] = np.log10(y["temperature"])

    # Standardize the data column-wise
    X_standardized = StandardScaler().fit_transform(X)

    return X_standardized, y


@st.cache_data
def dimensionality_reduction(data, dimension_reduction_algo, n_dimension):
    """
    Perform dimensionality reduction on the dataset.
    dimension_reduction_algo: Algorithm to use for dimensionality reduction.
    n_dimension: Number of dimensions to reduce to.

    Returns the reduced dataset.
    """
    unsupervised_ml = UnsupervisedML()
    if dimension_reduction_algo == "PCA":
        data_reduced = unsupervised_ml.reduce_dataset_pca(
            data, n_components=n_dimension
        )
    elif dimension_reduction_algo == "tSNE":
        data_reduced = unsupervised_ml.reduce_dataset_tsne(
            data, n_components=n_dimension
        )
    return data_reduced


######### Setup Streamlit app UI ################
st.title("TelescopeML Unsupervised Learning")
unsupervised_ml = UnsupervisedML()

# Generate some sample data
data, output_variables = import_data()

# Sidebar for selecting dimension reduction and clustering algorithms
dimension_reduction_algo = st.sidebar.selectbox(
    "Choose Dimension Reduction Method", ["PCA", "tSNE"]
)
n_dimension = st.sidebar.selectbox("Choose # of Dimensions", ["2", "3"])
n_dimension = int(n_dimension)
clustering_algo = st.sidebar.selectbox(
    "Choose Clustering Method", ["KMeans", "DBSCAN", "NONE"]
)

data_reduced = dimensionality_reduction(data, dimension_reduction_algo, n_dimension)

######### Allow User to select clustering algorithm with custom parameters ################
if clustering_algo == "KMeans":
    # Select number of clusters using a slider
    n_clusters = st.slider("Select Number of Clusters", 2, 20, 3)

    # Select method of initialization
    init = st.selectbox(
        "Method for Initialization",
        ["k-means++", "random"],
        help="""k-means++ : selects initial cluster centroids using sampling based on
        an empirical probability distribution of the points' contribution to the
        overall inertia\n
random: choose observations (rows) at random from data for the initial centroids\n""",
    )

    kmeans = unsupervised_ml.kmeans_clustering(
        data_reduced, n_clusters=n_clusters, init=init
    )
    labels, centers = kmeans.labels_, kmeans.cluster_centers_

    plt = unsupervised_ml.plot_kmeans_clusters(data_reduced, labels, centers)
elif clustering_algo == "DBSCAN":
    # Select DBSCAN parameters
    eps = st.slider(
        "Select Epsilon",
        0.1,
        2.0,
        0.5,
        0.1,
        help="Maximum distance between two samples for them to be considered in the same neighborhood",
    )
    min_samples = st.slider(
        "Select Minimum Samples",
        2,
        20,
        5,
        help="Number of samples in a neighborhood for a point to be considered a core point",
    )

    # Perform DBSCAN clustering
    dbscan = unsupervised_ml.dbscan_clustering(
        data_reduced, eps=eps, min_samples=min_samples
    )

    labels = dbscan.labels_

    # Print number of clusters
    n_clusters = len(np.unique(labels)) - 1
    st.write(f"Number of Clusters: {n_clusters}")

    plt = unsupervised_ml.plot_dbscan_clusters(
        data_reduced, labels, dbscan.core_sample_indices_, labels == -1
    )
elif clustering_algo == "NONE":
    output_var = st.sidebar.selectbox(
        "Select Output Overlay Option",
        ["gravity", "temperature", "c_o_ratio", "metallicity", "NONE"],
    )

    output_labels = None
    if output_var != "NONE":
        output_labels = output_variables[output_var]

    plt = unsupervised_ml.plot_dimension_reduction(
        data_reduced, dimension_reduction_algo, output_labels, output_var
    )

######### Plot Results ################
st.pyplot(plt)

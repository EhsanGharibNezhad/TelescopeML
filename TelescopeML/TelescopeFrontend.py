# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from Unsupervised_ML import perform_pca, perform_clustering  # Importing from the separate module

# Set page configuration
st.set_page_config(page_title="TelescopeML Interaction", layout="wide")


##############
#  Set up the data
##############

# Fetch the reference data path from environment variable
__reference_data_path__ = os.getenv("TelescopeML_reference_data")


def load_data():
    """Load the training dataset."""
    data_path = os.path.join(
        __reference_data_path__,
        'training_datasets',
        'browndwarf_R100_v4_newWL_v3.csv.bz2'
    )
    return pd.read_csv(data_path, compression='bz2')

# Load the dataset
train_BD = load_data()

# Define output and wavelength feature names
output_names = ['gravity', 'temperature', 'c_o_ratio', 'metallicity']
wavelength_names = [item for item in train_BD.columns if item not in output_names]

# Training variables
X = train_BD.drop(columns=output_names)

# Target/Output feature variables
y = train_BD[output_names]
y['temperature'] = np.log10(y['temperature'])

########################
# Title and description
######################

st.title('Telescope ML Interactive Experience')
st.write('This StreamLit allows you to experiment with unsupervised learning methods on a dataset of readings from brown dwarf planets.')

st.subheader('How It Works')
st.write("- A principal component is a a vector in the input space that points in the directions where the data is most spread out")
st.write("- You select how many dimensions you'd like to view the data in, then PCA finds that many directions to capture the most variance.")         

############
# PCA
############

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

# Select number of principal components to graph
n_components = st.radio('Select number of principal components', [2, 3])

# Perform PCA using the separate module
pca, components = perform_pca(data_scaled, n_components)

# Create a DataFrame with the principal components
components_df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(n_components)])

# Calculate total variance captured
total_variance = np.sum(pca.explained_variance_ratio_)

# --- Combined Variance and Principal Components Section ---
st.subheader('Variance and Principal Components')

# Create two columns: one for variance information and one for principal components
variance_col, pc_col = st.columns(2)

with variance_col:
    st.markdown("**Explained Variance Information:**")
    explained_variance_df = pd.DataFrame({
        'Principal Component': [f'PC{i+1}' for i in range(n_components)] + ['Total Variance'],
        'Variance Ratio': list(pca.explained_variance_ratio_) + [total_variance]
    })
    explained_variance_df['Variance Ratio'] = explained_variance_df['Variance Ratio'].apply(lambda x: f"{x:.2%}")

    # Reset the index and drop it to mimic hiding
    explained_variance_display = explained_variance_df.reset_index(drop=True)

    # Display the table without the index
    st.table(explained_variance_display)

with pc_col:
    st.markdown("**Principal Components Weights:**")
    
    pc_df = pd.DataFrame(pca.components_, columns=X.columns, index=[f'PC{i+1}' for i in range(n_components)])

    
    styled_pc_df = pc_df.style.format("{:.4f}")


    st.dataframe(styled_pc_df)
    
st.write("- Now that the data is in 2 or 3 dimensions so you can understand it you can color it according to labels like temperature (on the left) or you can try to cluster it (on the right)")

# --- Coloring Options ---
st.subheader('Coloring Options')

# Create two columns for coloring options
coloring_col1, coloring_col2 = st.columns(2)

# ---- Coloring by Target Variable ----
with coloring_col1:
    st.markdown("**Color by Target Variable**")
    label_choice = st.selectbox(
        'Select a target variable for coloring', 
        ['None'] + y.columns.tolist(), 
        key='label_color'
    )
    if label_choice != 'None':
        components_df['Label'] = y[label_choice].values
    else:
        components_df['Label'] = 'All'

# ---- Coloring by Clustering Method ----
with coloring_col2:
    st.markdown("**Color by Clustering**")

    selected_algo = st.selectbox('Select Clustering Algorithm', ['K-Means', 'DBSCAN', 'Gaussian Mixture'], key='clustering_algorithm')

    # Clustering Parameters
    if selected_algo == 'K-Means':
        n_clusters = st.slider('Select number of clusters', 2, 10, 3, key='kmeans_clusters')
        cluster_labels, cluster_centers = perform_clustering(components, 'K-Means', {'n_clusters': n_clusters})

    elif selected_algo == 'DBSCAN':
        eps = st.slider('Select epsilon (eps)', 0.1, 5.0, 0.5, step=0.1, key='dbscan_eps')
        min_samples = st.slider('Select minimum samples', 1, 10, 5, key='dbscan_min_samples')
        cluster_labels, cluster_centers = perform_clustering(components, 'DBSCAN', {'eps': eps, 'min_samples': min_samples})

        if -1 in cluster_labels:
            st.warning('DBSCAN identified noise points which are labeled as -1.')
            cluster_labels = np.where(cluster_labels == -1, 'Noise', cluster_labels.astype(str))

    elif selected_algo == 'Gaussian Mixture':
        n_components_gmm = st.slider('Select number of components', 2, 10, 3, key='gmm_components')
        cluster_labels, cluster_centers = perform_clustering(components, 'Gaussian Mixture', {'n_components': n_components_gmm})

    components_df['Cluster'] = cluster_labels.astype(str)

# --- Plotting Section ---
# Create two columns for the plots
plot_col1, plot_col2 = st.columns(2)

# ---- Helper Function to Create PCA Plots ----
def create_pca_plot(df, color, title_suffix):
    if n_components == 2:
        fig = px.scatter(
            df,
            x='PC1',
            y='PC2',
            color=color,
            title=f'PCA Colored by {title_suffix}',
            opacity=0.7,
            hover_data=df.columns,
            color_continuous_scale='Viridis'
        )
    else:
        fig = px.scatter_3d(
            df,
            x='PC1',
            y='PC2',
            z='PC3',
            color=color,
            title=f'3D PCA Colored by {title_suffix}',
            opacity=0.7,
            hover_data=df.columns,
            color_continuous_scale='Viridis'
        )
        fig.update_traces(marker=dict(size=3))
    fig.update_layout(legend_title_text=color)
    return fig

# ---- Plot Colored by Label ----
with plot_col1:
    st.subheader('PCA Plot - Colored by Label')
    fig1 = create_pca_plot(components_df, 'Label', 'Label')
    st.plotly_chart(fig1, use_container_width=True)

# ---- Plot Colored by Clustering ----
with plot_col2:
    st.subheader('PCA Plot - Colored by Clustering')
    fig2 = create_pca_plot(components_df, 'Cluster', 'Cluster')

    # Plot cluster centers if applicable
    if cluster_centers is not None:
        centers_df = pd.DataFrame(cluster_centers, columns=[f'PC{i+1}' for i in range(n_components)])
        if n_components == 2:
            centers_plot = px.scatter(
                centers_df, 
                x='PC1', 
                y='PC2'
            )
            centers_trace = centers_plot.update_traces(
                mode='markers',
                marker=dict(size=12, symbol='x', color='red', line=dict(width=2, color='DarkSlateGrey')),
                name='Cluster Centers'
            ).data[0]
            fig2.add_trace(centers_trace)
        else:
            centers_plot = px.scatter_3d(
                centers_df, 
                x='PC1', 
                y='PC2', 
                z='PC3'
            )
            centers_trace = centers_plot.update_traces(
                mode='markers',
                marker=dict(size=5, symbol='x', color='red', line=dict(width=2, color='DarkSlateGrey')),
                name='Cluster Centers'
            ).data[0]
            fig2.add_trace(centers_trace)
    st.plotly_chart(fig2, use_container_width=True)

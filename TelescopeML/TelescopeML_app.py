import streamlit as st
import pandas as pd
import bz2
import io
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from unsupervised_ml import TrainMlUnsupervised  
import time 

# -----------------------------------------------------
# Custom styles for the app using HTML and CSS
# -----------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #FFFFFF; 
        color: #2F4F4F; 
    }
    .stButton>button {
        background-color: #007BFF; 
        color: black; 
        border-radius: 10px; 
        border: none; 
    }
    .stButton>button:hover {
        background-color: #0056b3; 
        color: black; 
    }
    .stSlider>div { 
        color: #2F4F4F; 
    }
    .stSelectbox>div { 
        color: #829595; 
    }
    .stSelectbox select { 
        background-color: #555555; 
        color: white; 
        border-radius: 5px; 
        padding: 5px; 
        border: 1px solid #555555; 
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------
# App Title and Description
# -----------------------------------------------------
st.markdown("<h1 style='text-align: center; color: #000000;'>Enhancing TelescopeML With Unsupervised Learning</h1>", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; color: #333333;'>
        Welcome to the TelescopeML: Unsupervised Learning Explorer!<br>
        Upload your dataset and explore various unsupervised machine learning techniques such as PCA, K-Means Clustering, DBSCAN, and Anomaly Detection.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------
# File Uploader with Instructions
# -----------------------------------------------------

st.markdown("""
    <h2 style='font-size: 24px; font-weight: bold; color: #333333;'>Upload Your Dataset</h2>
    """, unsafe_allow_html=True)

st.markdown("""
    Upload your dataset in CSV format or as a bz2-compressed CSV file. 
    Ensure that the file size does not exceed 200MB.
""")

uploaded_file = st.file_uploader(
    "Choose a CSV file or bz2 compressed file",
    type=["csv", "bz2"],
    help="Upload your dataset in CSV format or as a bz2-compressed CSV. Maximum file size: 200MB."
)

# -----------------------------------------------------
# Function to Load and Preprocess Data
# -----------------------------------------------------
def load_data(file):
    """
    Load data from an uploaded file, handling CSV and bz2 compressed formats.

    Parameters:
    - file: Uploaded file object from Streamlit's file_uploader.

    Returns:
    - df: Pandas DataFrame containing the loaded and preprocessed data.
    """
    try:
        if file.name.endswith('.bz2'):
            with bz2.BZ2File(file, 'rb') as bz2_file:
                data = bz2_file.read()  
                df = pd.read_csv(io.BytesIO(data))  
        else:
            df = pd.read_csv(file)

        ml_model = TrainMlUnsupervised()

        df = ml_model.load_and_preprocess_data(df)

        return df  
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None 

# -----------------------------------------------------
# Data Preview
# -----------------------------------------------------
if uploaded_file is not None:
    df = load_data(uploaded_file) 

    if df is not None:
        st.write(f'**Loaded Data (showing first 5 rows):**')
        st.write(df.head())

        st.write(f'**Initial Data Shape:** {df.shape}')

        feature_values = df.values 
        feature_names = df.columns.tolist() 

        scaler = StandardScaler()
        feature_values_standardized = scaler.fit_transform(feature_values)  

        feature_df = pd.DataFrame(feature_values_standardized, columns=feature_names)

        # -----------------------------------------------------
        # Feature Selection for Visualization
        # -----------------------------------------------------
        st.markdown("""
            <h2 style='font-size: 24px; font-weight: bold; color: #333333;'>Select X and Y Axes for Visualization</h2>
            """, unsafe_allow_html=True)

        st.markdown("""
            Choose the features you want to visualize on the X and Y axes of the scatter plot. 
            This will help you explore the relationship between different variables in your dataset.
        """)
        x_axis = st.selectbox('Select X-axis:', options=feature_names, help="Choose a feature for the X-axis of the scatter plot.")
        y_axis = st.selectbox('Select Y-axis:', options=feature_names, help="Choose a feature for the Y-axis of the scatter plot.")

        # -----------------------------------------------------
        # Method Selection and Parameter Configuration
        # -----------------------------------------------------
        st.markdown("""
            <h2 style='font-size: 24px; font-weight: bold; color: #333333;'>Clustering and Dimensionality Reduction Methods</h2>
            """, unsafe_allow_html=True)
        st.markdown("""
            Select the unsupervised learning method you want to apply to your data. 
            Each method has its own parameters that you can adjust to tailor the analysis to your needs.
        """)
        method = st.selectbox('Choose Method', ['Choose Method', 'K-Means', 'PCA', 'DBSCAN', 'Anomaly Detection'], key='method_select')

        if method == 'K-Means':
            st.markdown("""
                **K-Means Clustering** partitions the data into K distinct clusters based on feature similarity.
                Adjust the number of clusters to see how the data is grouped.
            """)
            n_clusters = st.slider('Number of Clusters', min_value=2, max_value=10, value=3, help="Specify the number of clusters for K-Means.")
        elif method == 'PCA':
            st.markdown("""
                **Principal Component Analysis (PCA)** reduces the dimensionality of the data while retaining most of the variance.
                Choose the number of principal components to keep for visualization.
            """)
            n_components = st.slider('Number of Components', min_value=2, max_value=min(df.shape[1], 3), value=2, help="Choose the number of principal components to retain.")
        elif method == 'DBSCAN':
            st.markdown("""
                **DBSCAN** identifies clusters based on density and can find arbitrarily shaped clusters.
                Adjust the epsilon (eps) and minimum samples parameters to control cluster formation.
            """)
            eps = st.slider('Epsilon (eps)', min_value=0.1, max_value=10.0, value=0.5, step=0.1, help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
            min_samples = st.slider('Minimum Samples', min_value=1, max_value=20, value=5, help="The number of samples in a neighborhood for a point to be considered as a core point.")
        elif method == 'Anomaly Detection':
            st.markdown("""
                **Anomaly Detection** helps identify data points that do not fit the general pattern of the dataset.
                Choose the method you want to use for anomaly detection and select features to visualize.
            """)
            anomaly_method = st.selectbox('Choose Anomaly Detection Method', ['Isolation Forest', 'Local Outlier Factor'], key='anomaly_method_select')

            if anomaly_method == 'Isolation Forest':
                st.markdown("""
                    **Isolation Forest** is an anomaly detection method that isolates anomalies instead of profiling normal data points.
                """)
                contamination = st.slider('Contamination Ratio', min_value=0.01, max_value=0.5, value=0.1, step=0.01, help="Proportion of outliers in the data set.")
            elif anomaly_method == 'Local Outlier Factor':
                st.markdown("""
                    **Local Outlier Factor (LOF)** detects anomalies based on the local density of data points.
                """)
                n_neighbors = st.slider('Number of Neighbors', min_value=5, max_value=50, value=20, help="Number of neighbors to use for anomaly detection.")

        # -----------------------------------------------------
        # Perform Selected Method
        # -----------------------------------------------------
        if st.button('Run Method'):
            if method == 'K-Means':
                st.write(f"**Running K-Means with {n_clusters} Clusters**")
                kmeans = KMeans(n_clusters=n_clusters)
                clusters = kmeans.fit_predict(feature_values_standardized)
                feature_df['Cluster'] = clusters

                # Visualization of K-Means clustering
                fig = px.scatter(feature_df, x=x_axis, y=y_axis, color='Cluster', title=f'K-Means Clustering with {n_clusters} Clusters')
                st.plotly_chart(fig)

            elif method == 'PCA':
                st.write(f"**Running PCA with {n_components} Components**")
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(feature_values_standardized)
                pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
                
                # Visualization of PCA
                fig = px.scatter(pca_df, x='PC1', y='PC2', title=f'PCA with {n_components} Components')
                st.plotly_chart(fig)

            elif method == 'DBSCAN':
                st.write(f"**Running DBSCAN with eps={eps} and min_samples={min_samples}**")
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(feature_values_standardized)
                feature_df['Cluster'] = clusters

                # Visualization of DBSCAN clustering
                fig = px.scatter(feature_df, x=x_axis, y=y_axis, color='Cluster', title='DBSCAN Clustering')
                st.plotly_chart(fig)

            elif method == 'Anomaly Detection':
                if anomaly_method == 'Isolation Forest':
                    st.write(f"**Running Isolation Forest with contamination={contamination}**")
                    iso_forest = IsolationForest(contamination=contamination)
                    anomalies = iso_forest.fit_predict(feature_values_standardized)
                    feature_df['Anomaly'] = anomalies

                    # Visualization of anomalies
                    fig = px.scatter(feature_df, x=x_axis, y=y_axis, color='Anomaly', title='Anomaly Detection with Isolation Forest')
                    st.plotly_chart(fig)

                elif anomaly_method == 'Local Outlier Factor':
                    st.write(f"**Running Local Outlier Factor with n_neighbors={n_neighbors}**")
                    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
                    anomalies = lof.fit_predict(feature_values_standardized)
                    feature_df['Anomaly'] = anomalies

                    # Visualization of anomalies
                    fig = px.scatter(feature_df, x=x_axis, y=y_axis, color='Anomaly', title='Anomaly Detection with Local Outlier Factor')
                    st.plotly_chart(fig)

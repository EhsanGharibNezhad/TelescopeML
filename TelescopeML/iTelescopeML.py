#Interactive Platform for TelescopeML and Unsupervised_ML.py
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from TelescopeML.DataMaster import *
from TelescopeML.DeepTrainer import *
from TelescopeML.Predictor import *
from TelescopeML.IO_utils import load_or_dump_trained_model_CNN
from TelescopeML.StatVisAnalyzer import *
import os
import Unsupervised_ML
import streamlit as st
__reference_data__ = os.getenv("TelescopeML_reference_data")
__reference_data_path__ = __reference_data__



###############################################################################
############################## Utility Functions ##############################
###############################################################################

def load_spectra_data():
    """
    Executes the functionality to load and prepare (standardize & feature engineer ) the spectra data from the notebooks.
    
    """
    #Load the data
    train_BD = pd.read_csv(os.path.join(__reference_data_path__, 'training_datasets', 'browndwarf_R100_v4_newWL_v3.csv.bz2'), compression='bz2')
    output_names = ['gravity', 'temperature', 'c_o_ratio', 'metallicity']
    wavelength_names = [item for item in train_BD.columns.to_list() if item not in output_names]
    wavelength_values = [float(item) for item in wavelength_names]
    #Prepare the training features X and target variables y for ML models
    # Training  variables
    X = train_BD.drop(
        columns=['gravity', 
                 'temperature', 
                 'c_o_ratio', 
                 'metallicity'])
    # Target/Output feature variables
    y = train_BD[['gravity', 'c_o_ratio', 'metallicity', 'temperature', ]]
    #Apply a log transform to the temperature feature since this is what is recommended by the Notebooks
    y.loc[:, 'temperature'] = np.log10(y['temperature'])
    
    ###############################################################################
    #Using the DataMaster.py module, load the X and y dataframes into a DataProcessor object which has built-in engineering functionality
    data_processor = DataProcessor(flux_values=X.to_numpy(),
                                   wavelength_names=X.columns,
                                   wavelength_values=wavelength_values,
                                   output_values=y.to_numpy(),
                                   output_names=output_names,
                                   spectral_resolution=200)

    #Set up training, testing, and validation datasets
    data_processor.split_train_validation_test(test_size=0.1, val_size=0.1, random_state_=42)
    # Scale (standardize) the X features using MinMax Scaler
    data_processor.standardize_X_row_wise()
    # Standardize the y features using Standard Scaler
    data_processor.standardize_y_column_wise()
    
    #Apply the same feature engineering to take the min and max of each row
    # train
    data_processor.X_train_min = data_processor.X_train.min(axis=1)
    data_processor.X_train_max = data_processor.X_train.max(axis=1)
    # validation
    data_processor.X_val_min = data_processor.X_val.min(axis=1)
    data_processor.X_val_max = data_processor.X_val.max(axis=1)
    # test
    data_processor.X_test_min = data_processor.X_test.min(axis=1)
    data_processor.X_test_max = data_processor.X_test.max(axis=1)

    df_MinMax_train = pd.DataFrame((data_processor.X_train_min, data_processor.X_train_max)).T
    df_MinMax_val = pd.DataFrame((data_processor.X_val_min, data_processor.X_val_max)).T
    df_MinMax_test = pd.DataFrame((data_processor.X_test_min, data_processor.X_test_max)).T

    df_MinMax_train.rename(columns={0:'min', 1:'max'}, inplace=True)
    df_MinMax_val.rename(columns={0:'min', 1:'max'}, inplace=True)
    df_MinMax_test.rename(columns={0:'min', 1:'max'}, inplace=True)
    
    #Now Standardize the min and max features
    data_processor.standardize_X_column_wise(output_indicator='Trained_StandardScaler_X_ColWise_MinMax',
                                            X_train = df_MinMax_train.to_numpy(),
                                            X_val = df_MinMax_val.to_numpy(),
                                            X_test = df_MinMax_test.to_numpy())

    #Concatenate all of the features into training, testing, and validation dataframes
    X_train_standardized = pd.concat([pd.DataFrame(data_processor.X_train_standardized_rowwise), pd.DataFrame(data_processor.X_train_standardized_columnwise)],axis=1)
    X_val_standardized = pd.concat([pd.DataFrame(data_processor.X_val_standardized_rowwise), pd.DataFrame(data_processor.X_val_standardized_columnwise)],axis=1)
    X_test_standardized = pd.concat([pd.DataFrame(data_processor.X_test_standardized_rowwise), pd.DataFrame(data_processor.X_test_standardized_columnwise)],axis=1)
    return data_processor, X_train_standardized, X_val_standardized, X_test_standardized



###############################################################################
############################### iTelescopeML.py ###############################
###############################################################################
st.image('Brown_Dwarf.PNG') #Check the path to this if getting errors

# Set the title of the app
st.title("iTelescopeML (beta)")

# Add a header
st.header("Overview of iTelescopeML")

# Add some introductory text
st.write("Welcome to iTelescopeML! This is the baseline of a simple platform that allows users to utilize TelescopeMLs capabilities in an interavtive way.")
st.write('')
st.write('In this current beta version, this platform only offers a few capabilities:')
st.write('- Load and prepare the brown dwarf spectra dataset (the same that is utilized in the TelescopeML tutorials).')
st.write('- Instantiate an UnsupervisedML object from the newly developed Unsupervised_ML.py module.')
st.write('- One can then perform a series of unsupervised machine learning tasks including dimensionality reduction (PCA and t-SNE embeddings) and clustering (k-means, DBSCAN, and spectral clustering).')
st.write('')
st.write('Future work: Finish integrating all TelescopeML capabilities onto the platform as well as develop a smoother user interface')
st.write('(This beta is simply meant to serve as a framework for future development)')


#########################################################
# header for section "Loading and Preping Dataset"
st.header("Loading and Preparing the Brown Dwarf Spectra Dataset")
st.write('This section provides a quick and simple way to load the post-processessed form of the brown dwarf spectra dataset (from the TelescopeML tutorials)')

# Button that loads data
if 'load_data_button_clicked' not in st.session_state:
    st.session_state.load_data_button_clicked = False
if 'load_data_button_ran' not in st.session_state:
    st.session_state.load_data_button_ran = False
if st.button('Load & Prepare Spectra Data'):
    st.session_state.load_data_button_clicked = True
if st.session_state.load_data_button_clicked == True and st.session_state.load_data_button_ran == False:
    #Load the data
    data_processor, X_train_standardized, X_val_standardized, X_test_standardized = load_spectra_data()
    #Initialize session state variables so that they persist past this button press
    st.session_state.data_processor = data_processor
    st.session_state.X_train_standardized = X_train_standardized
    st.session_state.X_val_standardized = X_val_standardized
    st.session_state.X_test_standardized = X_test_standardized
    st.write("Finished loading and preparing dataset.")
    st.write(f"Shape of training dataset: {X_train_standardized.shape}.")
    st.session_state.load_data_button_ran = True




#########################################################
# header for section "Unsupervised Learning"
st.header("Unsupervised Learning with TelescopeML")
st.write("This section will generate an instantiation of a UnsupervisedML object from the newly developed Unsupervised_ML module. This UML object will store all of the different embeddings and clusterings that we generate from our data.")
#Instantiate an UnsupervisedML object
if 'UML_button_clicked' not in st.session_state:
    st.session_state.UML_button_clicked = False
if 'UML_button_ran' not in st.session_state:
    st.session_state.UML_button_ran = False
UML_button_disabled = not st.session_state.load_data_button_clicked
if st.button('Instantiate UML Object', disabled=UML_button_disabled):
    st.session_state.UML_button_clicked = True
if st.session_state.UML_button_clicked == True and st.session_state.UML_button_ran == False:
    UML = Unsupervised_ML.UnsupervisedML(st.session_state.X_train_standardized)
    st.session_state.UML = UML
    st.write("Finished instantiating UnsupervisedML object.")
    st.session_state.UML_button_ran = True


#########################################################
st.subheader("Dimensionality Reduction Techniques")
st.write("This section allows the user to generate two types of embeddings of the wavelength data into a lower-dimension space: principal components (PCA) or t-distributed stochastic neighbors (t-SNE).")
st.write("- PCA: A classic way of reducicing the feature space of a dataset by finding the vectors of maximum variance for the dataset. This is a good way to embedd your data into a lower dimension space while maintaining a high amount of variance that is explained by your dataset.")
st.write("- t-SNE: A more advanced embedding technique that can typically capture better patterns than just PCA. However, it is recommended to perform t-SNE on the already reduced space generated by PCA (use 10 PCs). This will help reduce the amount of computation needed for t-SNE while still retaining the majority of the variance in the original dataset.")
st.warning("Training a t-SNE embedding may take between 5-10 minutes.")

if 'PCA_button_clicked' not in st.session_state:
    st.session_state.PCA_button_clicked = False
if 'PCA_button_ran' not in st.session_state:
    st.session_state.PCA_button_ran = False

col1, col2 = st.columns(2)
with col1:
    PCA_button_disabled = not st.session_state.UML_button_clicked
    if st.button('PCA', disabled=PCA_button_disabled):
        st.session_state.PCA_button_clicked = not st.session_state.PCA_button_clicked
        if st.session_state.PCA_button_clicked == False:
            st.session_state.PCA_button_ran = False
    if st.session_state.PCA_button_clicked == True and st.session_state.PCA_button_ran == False:
        n_components = st.selectbox("Choose the number of principal components to keep", ['Select an option...',3,4,5,6,7,8,9,10], key='PCA_selectbox')
        if st.session_state.PCA_button_clicked == True and n_components != 'Select an option...':
            #Generate the PCA embeddings
            st.session_state.UML.pca(train_data=st.session_state.UML.dataset, val_data=None, test_data=None, n_components=10, random_state=42)
            st.write("Finished PCA embedding.")
            st.session_state.PCA_button_ran = True
        
if 'TSNE_button_clicked' not in st.session_state:
    st.session_state.TSNE_button_clicked = False
if 'TSNE_button_ran' not in st.session_state:
    st.session_state.TSNE_button_ran = False
    
with col2:    
    TSNE_button_disabled = not st.session_state.PCA_button_clicked or not st.session_state.UML_button_clicked
    if st.button('t-SNE', disabled=TSNE_button_disabled):
        st.session_state.TSNE_button_clicked = not st.session_state.TSNE_button_clicked
        if st.session_state.TSNE_button_clicked == False:
            st.session_state.TSNE_button_ran = False
    if st.session_state.TSNE_button_clicked == True and st.session_state.TSNE_button_ran == False:
        n_components = st.selectbox("Choose the number of dimensions of the embedded space", ['Select an option...',3], key='TSNE_selectbox_1')
        learning_rate = st.selectbox("Choose the learning rate", ['Select an option...','auto'], key='TSNE_selectbox_2')
        perplexity = st.selectbox("Choose the perplexity", ['Select an option...',5,50,100,200], key='TSNE_selectbox_3')
        if st.session_state.TSNE_button_clicked == True and n_components != 'Select an option...' and learning_rate != 'Select an option...' and perplexity != 'Select an option...':
            #Generate t-SNE embeddings
            st.session_state.UML.tsne(train_data=st.session_state.UML.pca_train_dataset, val_data=None, test_data=None, n_components = n_components, learning_rate=learning_rate, perplexity=perplexity, random_state=42)
            st.write("Finished t-SNE embedding.")
            st.session_state.TSNE_button_ran = True



#########################################################
st.subheader("Clustering Techniques")
st.write("This section allows the user to generate clusters of the newly embedded dataset via thre different clustering techniques.")
st.write("- K-Means: The most classic clustering algorithm for determining clusters based on simple euclidean distances. Good safe overall starting choice.")
st.write("- DBSCAN: Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is a technique that is good for datasets with noise but has clusters of similar density (honestly, not the best choice for this dataset, as will be seen).")
st.write("- Spectral Clustering: A very strong technique that performs very well when the structure of the clusters is highly non-convex (as they are in this dataset).")


col1, col2 = st.columns(2)
with col1:
    if 'KMeans_PCA_button_clicked' not in st.session_state:
        st.session_state.KMeans_PCA_button_clicked = False
    if 'KMeans_PCA_button_ran' not in st.session_state:
        st.session_state.KMeans_PCA_button_ran = False
    KMeans_PCA_button_disabled = not st.session_state.UML_button_clicked or not st.session_state.PCA_button_clicked
    if st.button('K-Means on PCA', disabled=KMeans_PCA_button_disabled):
        st.session_state.KMeans_PCA_button_clicked = not st.session_state.KMeans_PCA_button_clicked
        if st.session_state.KMeans_PCA_button_clicked == False:
            st.session_state.KMeans_PCA_button_ran = False
    if st.session_state.KMeans_PCA_button_clicked == True and st.session_state.KMeans_PCA_button_ran == False:
        n_clusters = st.selectbox("Choose the number of clusters", ['Select an option...',3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], key='KMeans_PCA_selectbox')
        if st.session_state.KMeans_PCA_button_clicked == True and n_clusters != 'Select an option...':
            #Generate t-SNE embeddings
            st.session_state.UML.kmeans(train_data=st.session_state.UML.pca_train_dataset, val_data=None, test_data=None, n_clusters = n_clusters, n_init = 50, max_iter=500, random_state=42)
            st.write("Finished clustering.")
            fig = st.session_state.UML.scatter_3d_cluster_plot(dataset=st.session_state.UML.train_kmeans_clustered, method='PCA_KMeans')
            st.pyplot(fig)
            st.session_state.KMeans_PCA_button_ran = True

with col2:
    if 'KMeans_TSNE_button_clicked' not in st.session_state:
        st.session_state.KMeans_TSNE_button_clicked = False
    if 'KMeans_TSNE_button_ran' not in st.session_state:
        st.session_state.KMeans_TSNE_button_ran = False
    KMeans_TSNE_button_disabled = not st.session_state.UML_button_clicked or not st.session_state.TSNE_button_clicked
    if st.button('K-Means on t-SNE', disabled=KMeans_TSNE_button_disabled):
        st.session_state.KMeans_TSNE_button_clicked = not st.session_state.KMeans_TSNE_button_clicked
        if st.session_state.KMeans_TSNE_button_clicked == False:
            st.session_state.KMeans_TSNE_button_ran = False
    if st.session_state.KMeans_TSNE_button_clicked == True and st.session_state.KMeans_TSNE_button_ran == False:
        n_clusters = st.selectbox("Choose the number of clusters", ['Select an option...',3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], key='KMeans_TSNE_selectbox')
        if st.session_state.KMeans_TSNE_button_clicked == True and n_clusters != 'Select an option...':
            #Generate t-SNE embeddings
            st.session_state.UML.kmeans(train_data=st.session_state.UML.tsne_train_dataset, val_data=None, test_data=None, n_clusters = n_clusters, n_init = 50, max_iter=500, random_state=42)
            st.write("Finished clustering.")
            fig = st.session_state.UML.scatter_3d_cluster_plot(dataset=st.session_state.UML.train_kmeans_clustered, method='TSNE_KMeans')
            st.pyplot(fig)
            st.session_state.KMeans_TSNE_button_ran = True



col1, col2 = st.columns(2)
with col1:
    if 'DBSCAN_PCA_button_clicked' not in st.session_state:
        st.session_state.DBSCAN_PCA_button_clicked = False
    if 'DBSCAN_PCA_button_ran' not in st.session_state:
        st.session_state.DBSCAN_PCA_button_ran = False
    DBSCAN_PCA_button_disabled = not st.session_state.UML_button_clicked or not st.session_state.PCA_button_clicked
    if st.button('DBSCAN on PCA', disabled=DBSCAN_PCA_button_disabled):
        st.session_state.DBSCAN_PCA_button_clicked = not st.session_state.DBSCAN_PCA_button_clicked
        if st.session_state.DBSCAN_PCA_button_clicked == False:
            st.session_state.DBSCAN_PCA_button_ran = False
    if st.session_state.DBSCAN_PCA_button_clicked == True and st.session_state.DBSCAN_PCA_button_ran == False:
        eps = st.selectbox("Choose epsilon value (maximum distance between two samples for one to be considered as in the neighborhood of the other",
                           ['Select an option...', 0.1, 0.2, 0.5, 0.7, 1, 1.5, 2], key='DBSCAN_PCA_selectbox_1')
        min_samples = st.selectbox("Choose the number of samples (or total weight) in a neighborhood for a point to be considered as a core point",
                                   ['Select an option...', 1, 2, 5, 7, 10, 15], key='DBSCAN_PCA_selectbox_2')
        if st.session_state.DBSCAN_PCA_button_clicked == True and eps != 'Select an option...' and min_samples != 'Select an option...':
            #Generate t-SNE embeddings
            st.session_state.UML.dbscan(train_data=st.session_state.UML.pca_train_dataset, val_data=None, test_data=None, eps=eps, min_samples=min_samples, algorithm='brute', p=2)
            st.write("Finished clustering.")
            fig = st.session_state.UML.scatter_3d_cluster_plot(dataset=st.session_state.UML.train_dbscan_clustered, method='PCA_DBSCAN')
            st.pyplot(fig)
            st.session_state.DBSCAN_PCA_button_ran = True

with col2:
    if 'DBSCAN_TSNE_button_clicked' not in st.session_state:
        st.session_state.DBSCAN_TSNE_button_clicked = False
    if 'DBSCAN_TSNE_button_ran' not in st.session_state:
        st.session_state.DBSCAN_TSNE_button_ran = False
    DBSCAN_TSNE_button_disabled = not st.session_state.UML_button_clicked or not st.session_state.TSNE_button_clicked
    if st.button('DBSCAN on t-SNE', disabled=DBSCAN_TSNE_button_disabled):
        st.session_state.DBSCAN_TSNE_button_clicked = not st.session_state.DBSCAN_TSNE_button_clicked
        if st.session_state.DBSCAN_TSNE_button_clicked == False:
            st.session_state.DBSCAN_TSNE_button_ran = False
    if st.session_state.DBSCAN_TSNE_button_clicked == True and st.session_state.DBSCAN_TSNE_button_ran == False:
        eps = st.selectbox("Choose epsilon value (maximum distance between two samples for one to be considered as in the neighborhood of the other",
                           ['Select an option...', 0.1, 0.2, 0.5, 0.7, 1, 1.5, 2], key='DBSCAN_PCA_selectbox_1')
        min_samples = st.selectbox("Choose the number of samples (or total weight) in a neighborhood for a point to be considered as a core point",
                                   ['Select an option...', 1, 2, 5, 7, 10, 15], key='DBSCAN_PCA_selectbox_2')
        if st.session_state.DBSCAN_TSNE_button_clicked == True and eps != 'Select an option...' and min_samples != 'Select an option...':
            #Generate t-SNE embeddings
            st.session_state.UML.dbscan(train_data=st.session_state.UML.tsne_train_dataset, val_data=None, test_data=None, eps=eps, min_samples=min_samples, algorithm='brute', p=2)
            st.write("Finished clustering.")
            fig = st.session_state.UML.scatter_3d_cluster_plot(dataset=st.session_state.UML.train_dbscan_clustered, method='TSNE_DBSCAN')
            st.pyplot(fig)
            st.session_state.DBSCAN_TSNE_button_ran = True



st.warning("Generating spectral clusters may between 5-10 minutes.")

col1, col2 = st.columns(2)
with col1:
    if 'Spec_c_PCA_button_clicked' not in st.session_state:
        st.session_state.Spec_c_PCA_button_clicked = False
    if 'Spec_c_PCA_button_ran' not in st.session_state:
        st.session_state.Spec_c_PCA_button_ran = False
    Spec_c_PCA_button_disabled = not st.session_state.UML_button_clicked or not st.session_state.PCA_button_clicked
    if st.button('Spectral Clustering on PCA', disabled=Spec_c_PCA_button_disabled):
        st.session_state.Spec_c_PCA_button_clicked = not st.session_state.Spec_c_PCA_button_clicked
        if st.session_state.Spec_c_PCA_button_clicked == False:
            st.session_state.Spec_c_PCA_button_ran = False
    if st.session_state.Spec_c_PCA_button_clicked == True and st.session_state.Spec_c_PCA_button_ran == False:
        n_clusters = st.selectbox("Choose the number of clusters", ['Select an option...',3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], key='Spec_c_PCA_selectbox')
        if st.session_state.Spec_c_PCA_button_clicked == True and n_clusters != 'Select an option...':
            #Generate t-SNE embeddings
            st.session_state.UML.spec_c(train_data=st.session_state.UML.pca_train_dataset, affinity_metric='rbf', n_clusters=n_clusters, assign_labels="discretize", random_state=42)
            st.write("Finished clustering.")
            fig = st.session_state.UML.scatter_3d_cluster_plot(dataset=st.session_state.UML.train_spec_c_clustered, method='PCA_Spec_c')
            st.pyplot(fig)
            st.session_state.Spec_c_PCA_button_ran = True

with col2:
    if 'Spec_c_TSNE_button_clicked' not in st.session_state:
        st.session_state.Spec_c_TSNE_button_clicked = False
    if 'Spec_c_TSNE_button_ran' not in st.session_state:
        st.session_state.Spec_c_TSNE_button_ran = False
    Spec_c_TSNE_button_disabled = not st.session_state.UML_button_clicked or not st.session_state.PCA_button_clicked
    if st.button('Spectral Clustering on t-SNE', disabled=Spec_c_TSNE_button_disabled):
        st.session_state.Spec_c_TSNE_button_clicked = not st.session_state.Spec_c_TSNE_button_clicked
        if st.session_state.Spec_c_TSNE_button_clicked == False:
            st.session_state.Spec_c_TSNE_button_ran = False
    if st.session_state.Spec_c_TSNE_button_clicked == True and st.session_state.Spec_c_TSNE_button_ran == False:
        n_clusters = st.selectbox("Choose the number of clusters", ['Select an option...',3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], key='Spec_c_PCA_selectbox')
        if st.session_state.Spec_c_TSNE_button_clicked == True and n_clusters != 'Select an option...':
            #Generate t-SNE embeddings
            st.session_state.UML.spec_c(train_data=st.session_state.UML.tsne_train_dataset, affinity_metric='rbf', n_clusters=n_clusters, assign_labels="discretize", random_state=42)
            st.write("Finished clustering.")
            fig = st.session_state.UML.scatter_3d_cluster_plot(dataset=st.session_state.UML.train_spec_c_clustered, method='TSNE_Spec_c')
            st.pyplot(fig)
            st.session_state.Spec_c_TSNE_button_ran = True



st.subheader("Thank you for using iTelescopeML.py (beta)!")

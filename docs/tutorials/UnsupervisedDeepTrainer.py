# Author: Sierra Janson
# Unsupervised learning class and utility functions 

##################################################################
# Import Libraries: ##############################################
##################################################################
# other modules
from TelescopeML.DataMaster import *
# from io_funs import LoadSave

# data manipulation + statistical libraries
import pandas as pd
import numpy as np
from typing import Union

# data visualization
import matplotlib.pyplot as plt

# machine learning libraries
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Input,UpSampling1D
from tensorflow.keras.models import Model


##################################################################
# Utility Functions: #############################################
##################################################################
def load_data(datapath="TelescopeML_reference_data") -> DataProcessor:
    """Function to load data based on provided datapath environment variable.
    
    Parameters:
    - datapath (str): Name of datapath enviornment variable.

    Returns:
    - data_processor (DataProcessor): Object initialized with flux values and target variables.
    """
    import os 
    __reference_data_path__ = os.getenv(datapath)
    train_BD                = pd.read_csv(os.path.join(__reference_data_path__, 'training_datasets', 'browndwarf_R100_v4_newWL_v3.csv.bz2'), compression='bz2')

    output_names      = ['gravity', 'temperature', 'c_o_ratio', 'metallicity']
    wavelength_names  = [item for item in train_BD.columns.to_list() if item not in output_names]
    wavelength_values = [float(item) for item in wavelength_names]

    # initialize i/o for model
    X = train_BD.drop(columns=output_names)
    y = train_BD[['gravity', 'c_o_ratio', 'metallicity', 'temperature', ]]

    data_processor = DataProcessor( 
        flux_values=X.to_numpy(),
        wavelength_names=X.columns,
        wavelength_values=wavelength_values,
        output_values=y.to_numpy(),
        output_names=output_names,
        spectral_resolution=200,
        trained_ML_model=None,
        trained_ML_model_name='CNN',
    )
    return data_processor


def process_data(data_processor: DataProcessor) -> DataProcessor:
    """Extract min/max features from data and split into train, validation, and test datasets.

    Parameters:
    - data_processor (DataProcessor): Object storing the data to be process and feature extracted.

    Returns:
    - data_processor (DataProcessor): Object with processed data. 
    """
    data_processor.split_train_validation_test(test_size=0.1, val_size=0.1, random_state_=42,)
    data_processor.standardize_X_row_wise() 
    data_processor.standardize_y_column_wise()

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

    data_processor.standardize_X_column_wise(
        output_indicator='Trained_StandardScaler_X_ColWise_MinMax',
        X_train = df_MinMax_train.to_numpy(),
        X_val   = df_MinMax_val.to_numpy(),
        X_test  = df_MinMax_test.to_numpy(),
        )
    return data_processor


def df_formatter(df, atmospheric_parameters, new_label, new_values):
    """Appends atmospheric parameter columns to brown dwarf wavelength dataframe.
    
    Parameters:
    - df (pandas.DataFrame): Dataframe with flux values. 
    - ap (numpy.array): Array of target atmospheric values.
    
    Returns:
    - df (pandas.DataFrame): Dataframe appended with target atmospheric values.
    """
    df['gravity']      = atmospheric_parameters[:,[0]]
    df['c_o_ratio']    = atmospheric_parameters[:,[1]]
    df['metallicity']  = atmospheric_parameters[:,[2]]
    df['temperature']  = atmospheric_parameters[:,[3]]
    df[new_label]      = new_values
    return df

def reshape_x(X,dim):
    """Reshapes input dataset to desired dimension.
    
    Parameters:
    - X (numpy.array): Dataset for training.
    - dim (int): Desired Dimension.
    
    Returns:
    - (numpy.array): Reshaped training dataset.

    Raises:
    - (AssertError): If dimension provided was not 3 or 4.
    """
    if (dim == 3):
        return np.reshape(X, (1, X.shape[0],X.shape[1]))
    if (dim == 4):
        return np.reshape(X, (1, X.shape[0], X.shape[1], 1))
    assert(1 == 0 and "not a provided dimension")

class UnsupervisedML:
    """Class hosting various unsupervised methods to be utilized for training unsupervised flux data."""
    def __init__(self, X):
        """Initializes model.
        
        Parameters:
        - X (numpy.array): 104 main spectral features
        """
        self.X        = X       # 104 spectral features
        self.model    = None                                 
        self.history  = None
    
    def build_and_train_model(self, batch_size=32, epochs=1): # perhaps pass desired function to this
        """
        Build and fit the unsupervised CNN encoder model for feature extraction.

        Parameters:
            - batch_size (int): Number of batches for training.
            - epochs (int): Number of epochs for training.

        Returns:
            - history (REPLACE): Training history.
            - model (tf.keras.Model): Trained model.
        """

        #x_vals = self.X[:-1] if len(self.X)%2 != 0 else self.X
        x_vals = self.X[:5000]
        # Input shape object
        input_shape = Input(shape=(5000,104)) #25018

        # Encoding
        model     = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(input_shape)
        encoded     = MaxPooling1D(pool_size=4,padding='same')(model)
        # model     = Conv1D(128, 3, activation='relu', padding='same')(model)
        # encoded   = MaxPooling1D(2, padding='same')(model)

        model     = Conv1D(16, 3, activation='relu', padding='same')(encoded)
        model     = UpSampling1D(4)(model)
        # Decoding
        # model       = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(model)
        # model       = UpSampling1D(2)(model)
        decoded     = Conv1D(104,3, activation='sigmoid',padding='same')(model)

        # Initializing Model
        model = Model(input_shape, decoded)
        model.summary()
    
        # early_stop = EarlyStopping(monitor='loss', min_delta=4e-4, patience=50, mode='auto', \

        # Compile the model with an optimizer, loss function, and metrics
        # explain reasoning behind optimizer and loss function
        # explain preferences for initializing constants (GIN FILE/ outlined at top versus like this)

        # Run and compile Model
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(
            reshape_x(x_vals,3), 
            reshape_x(x_vals,3),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True
        )
        encoder       = Model(inputs=input_shape, outputs=encoded)
        self.model    = encoder
        self.history  = history

        return history, encoder
    

    def kmeans(self, features=[], k=3):
        """
        Clusters features into k categories.

        Parameters:
        - k (int): Discrete number of clusters.

        Returns:
        - labels (REPLACE): 1D array of cluster-labels  
        """
        from sklearn.cluster import KMeans

        #DELETE: ADD ELBOW GRAPH FUNCTION

        features  = self.X if len(features)==0 else features
        model     = KMeans(n_clusters=k)
        df        = pd.DataFrame(features)
        
        model.fit(df)
        return model.labels_ 
    
    def DBscan(self, max_radius, features=[], min_neighbors=10):
        """
        Cluster features into k categories.

        Parameters:
        - k (int): Discrete number of clusters.

        Returns:
        - labels (REPLACE): 1D array of cluster-labels  
        """
        from sklearn.cluster import DBSCAN

        #DELETE: is there a methodical way to determine parameters ?

        features  = self.X if len(features)==0 else features
        model     = DBSCAN(eps=max_radius, min_samples=min_neighbors)
        
        model.fit_predict(features)
        return model.labels_ 


    def PCA(self, keep_useful_proportion=0.99):
        """
        Reduces dimensionality of dataset while retaining most important features. 

        Parameters:
        - keep_useful_proportion (float): Proportion of useful features to keep

        Returns:
        - reduced_dataset (numpy.array): Dataset with reduced columns (reduced features)
        """
        from sklearn.decomposition import PCA

        pca              = PCA(keep_useful_proportion)
        reduced_dataset  = pca.fit_transform(self.X)
        return reduced_dataset 


    def plot_clusters(self,
            cluster_alg, 
            labels, # column of cluster labels
            df, # df of wavelengths
            ap=[], # atmospheric parameters
            xvar="c_o_ratio",
            yvar="temperature"
        ):
        """Plots clusters with respect to different variables.
        
        Parameters:
        - cluster_alg (str): "kmeans" or "dbscan" for desired clustering method to be plotted.
        - labels (numpy.array): 1D array of fitted cluster labels. 
        - df (pandas.DataFrame): Dataframe of flux values.
        - ap (numpy.array): Array of target atmospheric parameters.
        - xvar (str): X-axis variable desired to be analyzed in plot.
        - yvar (str): Y-axis variable desired to be analyzed in plot.
        """
        if len(ap != 0):
            df['gravity'] = ap[:,[0]]
            df['c_o_ratio'] = ap[:,[1]]
            df['metallicity'] = ap[:,[2]]
            df['temperature'] = ap[:,[3]]

        if cluster_alg == "kmeans":
            kmeans_df = df
            kmeans_df['kmeans'] = labels
            plt.scatter(x=kmeans_df[xvar],y=kmeans_df[yvar],c=kmeans_df['kmeans'])
            plt.gca().update(dict(title=f'{xvar} vs {yvar}', xlabel=xvar, ylabel=yvar))
            plt.colorbar()
            plt.show()
            return 

        if cluster_alg == "dbscan":
            dbscan_df = df
            dbscan_df['dbscan'] = labels
            plt.scatter(x=dbscan_df[xvar],y=dbscan_df[yvar],c=dbscan_df['dbscan'])
            plt.gca().update(dict(title=f'{xvar} vs {yvar}', xlabel=xvar, ylabel=yvar))
            plt.colorbar()
            plt.show()
            return
        print("Enter kmeans or dbscan as an option to plot.")



"""
Task:
- noise
- cleansing of wavelength data?
- load or save model 
- incorporate existing functions
"""
"""
What is the goal of the TelescopeML project
- accurate and swift determination of spectroscopic parameters (T_eff, log g, metallicity, radial v, macro/micro turbulence) from observational spectra

so why unsupervised learning??
--> unsupervised learning can't perform regression
you can cluster, reduce dimensionality, detect anomalies, and perform association rule learning (might work with ours)

clustering might unveil useful features to be passed to a regression model
anomaly detection & dataset reduction is definitely useful 

unsupervised learning methods for brown dwarfs would be ideal as it is difficult to obtain supervised data in this case

- pca, k-means, dbscan, cnn
https://medium.com/imagescv/top-8-most-important-unsupervised-machine-learning-algorithms-with-python-code-references-1222393b5077 
autoencoder, deep belief network, hierarchical temporal model, cnn, svms

- clustering may group certain absorption groups together which gives us information about the brown dwarfs' atmosphere
convolutional autoencoder


- three main tasks
- process synthetical astro ds for training a CNN model
- prepare obs ds for later use

- train CNN by implementing optimal hyperparameters (?)
- deploy CNN on actual data to derive spectroscopic parameters

--> train CNN with synthetic data, then classify observational data
    --> how is synthetic data formed? differences btwn obs + syn data? model performance?

extrasolar vs brown dwarf planets

(where do you see the future of Telescope ML?)
how does this compare to other open source analytical exoplanet libraries?
how will this library change with Roman?
why exoplanet atmospheric parameters?
(can we add a is it habitable function for fun)
cite all sources

"""

"""
Observations about existing project to implement:
- very well commented (even function parameters) & specific style
- specific docstring style for functions
- explicitly shows model architecture
- with example in docstrings
- explanatory variable names
- default parameters

Model:
- use tf
- optimizer, loss function, and metrics
- early stopping function
- add noise ? 
- return history, model
- scoring 


init, build, and fit function

Good practices:
- include typing
- declare constants at top of file
- modular code
"""
# File imports
import os

# Data Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# Plotting Imports
import matplotlib.pyplot as plt
import random

# PCA Imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

# Encoder-Decoder Imports
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam

# GMM Imports
from sklearn.mixture import GaussianMixture

__reference_data_path__ = os.getenv("TelescopeML_reference_data")
DEFAULT_ENCODER_PATH = os.path.join(__reference_data_path__, 'trained_ML_models/trained_weights_encoder_100epoch_20dimension.h5')
DEFAULT_DECODER_PATH = os.path.join(__reference_data_path__, 'trained_ML_models/trained_weights_decoder_100epoch_20dimension.h5')


class GMM:
    def __init__(self, X_train, X_val, X_test,
                       y_train, y_val, y_test):
        """
        Constructor for GMM class. X should be lower-dimension encoded space
        """
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test


    def build_model(self, config):
        """
        Build GMM model from parameters given in config
        """
        n_components = config['n_components']           # Cluster count
        covariance_type = config['covariance_type']     # full, tied, spherical, diag
        max_iter = config['max_iter']                   # Max EM steps to do
        n_init = config['n_init']                       # Number of initializations to make
        verbose = config['verbose']

        self.gmm = GaussianMixture(n_components, covariance_type=covariance_type, max_iter=max_iter, n_init=n_init, verbose=verbose)

    
    def calculate_expected_values(self):
        """
        Calculate expected values per cluster for an already-trained model
        """
        # use soft clustering to get average value for each cluster
        responsibilities = self.gmm.predict_proba(self.X_train)
        expected_outputs = np.zeros((self.gmm.n_components, self.y_train.shape[1]))

        # for each component, get weighted sum of outputs and divide by full responsibility to get averages
        for k in range(self.gmm.n_components):
            weighted_sum = np.dot(responsibilities[:, k], self.y_train)
            responsibility_sum = responsibilities[:, k].sum()
            expected_outputs[k, :] = weighted_sum / responsibility_sum

        # shape - (n_components, y_feature_count)
        self.expected_outputs = expected_outputs


    def train_model(self):
        """
        Fit GMM Model with existing train set, calculate & return expected outputs for each cluster
        """
        self.gmm.fit(self.X_train)

        return self.calculate_expected_values()
    

    def predict(self, X_input):
        """
        Given an X input, predict the 4 output features (grav, c/o, metal, temp)
        """
        responsibilities = self.gmm.predict_proba(X_input)
        estimated_Y = np.dot(responsibilities, self.expected_outputs)
        return estimated_Y


    def save_from_indicator(self, model_indicator: str):
        """
        Save model to reference data path given indicator
        """
        return self.save_to_path(os.path.join(__reference_data_path__, f'trained_ML_models/trained_gmm_{model_indicator}.pkl'))

    
    def load_from_indicator(self, model_indicator: str):
        """
        Load model from reference path given indicator
        """
        return self.load_from_path(os.path.join(__reference_data_path__, f'trained_ML_models/trained_gmm_{model_indicator}.pkl'))

    
    def save_to_path(self, path: str):
        """
        Save model to given path (using pickle object)
        """
        joblib.dump(self.gmm, path)
        return path


    def load_from_path(self, path: str):
        """
        Load GMM model from given path (pickle object)
        """
        self.gmm = joblib.load(path)
        return self.calculate_expected_values()



class Autoencoder:
    def __init__(self, X_train, X_val, X_test,
                       y_train, y_val, y_test):
        """
        Constructor for Autoencoder class. X data should be standardized columnwise.
        """
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test


    def build_model(self, config: dict):
        """
        Build autoencoder model from config hyperparameters
        encoder/decoder_layers must be a list of all layers in between input & output respectively.
        """
        input_dim = 104
        output_dim = config['output_dim']
        encoder_layers = config['encoder_layers']
        decoder_layers = config['decoder_layers']

        # construct encoder structure from config dict
        # with output size being input dimensions, and vice versa
        encoder_structure = [Dense(encoder_layers[0], activation='relu', input_shape=(input_dim,))]

        for i in range(1, len(encoder_layers)):
            encoder_structure.append(Dense(encoder_layers[i], activation='relu'))

        encoder_structure.append(Dense(output_dim, activation='linear'))

        # construct decoder structure from config dict
        # with input size being output dimensions, and vice versa
        decoder_structure = [Dense(decoder_layers[0], activation='relu', input_shape=(output_dim,))]

        for i in range(1, len(decoder_layers)):
            decoder_structure.append(Dense(decoder_layers[i], activation='relu'))

        decoder_structure.append(Dense(input_dim, activation='linear'))

        # construct encoder and decoder with previous layer lists
        encoder = Sequential(encoder_structure)
        decoder = Sequential(decoder_structure)

        # assign object variables to save encoder/decoder
        self.encoder = encoder
        self.decoder = decoder
        self.autoencoder = Model(inputs=self.encoder.input, outputs=self.decoder(self.encoder.output))

    
    def encode_input_data(self, dataset):
        """
        Return encoded input data (either using pre-existing set or new dataset)
        """
        key = {'train': self.X_train, 'val': self.X_val, 'test': self.X_test}
        
        # can either pass in own data of shape (x, 104) or use pre-existing set
        if type(key) == str and dataset in key:
            relevant_data = key[dataset]
        else:
            relevant_data = dataset

        return self.encoder.predict(relevant_data)
    

    def decode_output_data(self, encoded_data):
        """
        Reconstructs original 104D data from given encoded data
        """
        return self.decoder.predict(encoded_data)


    def train_model(self, epochs, batch_size, verbose):
        """
        Train autoencoder model with existing train set with specified parameters
        Returns tuple - (model_history, autoencoder). Doesn't return encoder or decoder alone
        """
        self.autoencoder.compile(loss='mse', optimizer=Adam())

        # model training
        model_history = self.autoencoder.fit(self.X_train, self.X_train,
                                             epochs=epochs, batch_size=batch_size, verbose=verbose,)

        return model_history, self.autoencoder
    

    def save_from_indicator(self, model_indicator: str):
        """
        Save encoder & decoder from model name/indicator, like '100epoch_20dimension'
        """
        self.save_weights(os.path.join(__reference_data_path__, f'trained_ML_models/trained_weights_encoder_{model_indicator}.h5'),
                          os.path.join(__reference_data_path__, f'trained_ML_models/trained_weights_decoder_{model_indicator}.h5'))
    

    def save_weights(self, encoder_weights_path, decoder_weights_path):
        """
        Save encoder & decoder weights to the given paths
        """
        self.encoder.save_weights(encoder_weights_path)
        self.decoder.save_weights(decoder_weights_path)


    def load_from_indicator(self, model_indicator):
        """
        Load encoder & decoder from model name/indicator, like '100epoch_20dimension'
        """
        self.load_weights(os.path.join(__reference_data_path__, f'trained_ML_models/trained_weights_encoder_{model_indicator}.h5'),
                          os.path.join(__reference_data_path__, f'trained_ML_models/trained_weights_decoder_{model_indicator}.h5'))

    
    def load_weights(self, encoder_weights_path = DEFAULT_ENCODER_PATH, decoder_weights_path = DEFAULT_DECODER_PATH):
        """
        Load encoder & decoder weights from given filepaths.
        """
        self.encoder.load_weights(encoder_weights_path)
        self.decoder.load_weights(decoder_weights_path)



class PCAModel:
    def __init__(self, X_train, X_val, X_test,
                       y_train, y_val, y_test):
        """
        Constructor for PCA class. X data should be standardized columnwise.
        """
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test


    def build_model(self, config: dict):
        """
        Build PCA model according to config dict
        """
        n_components = config['n_components']

        # n_components can either be > 1 (how many PCs to keep)
        #   or 0 < n < 1 (what % of variability to preserve)
        self.pca = PCA(n_components)

    def train_model(self):
        """
        Fit PCA Model with given dataset
        """
        self.pca.fit(self.X_train)


    def encode_input_data(self, dataset):
        """
        Return top output_dimensions PCs as encoded input data
        """
        key = {'train': self.X_train, 'val': self.X_val, 'test': self.X_test}
        
        # can either pass in own data of shape (x, 104) or use pre-existing set
        if type(key) == str and dataset in key:
            relevant_data = key[dataset]
        else:
            relevant_data = dataset
        
        return self.pca.transform(relevant_data)
    
    
    def decode_output_data(self, encoded_data):
        """
        Reconstructs original 104D data from encoded PCA data
        """
        return self.pca.inverse_transform(encoded_data)
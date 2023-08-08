# Import functions from other modules ============================
# from io_funs import LoadSave

# Import python libraries ========================================

# ******* Standard Data Manipulation / Statistical Libraries *****
import pandas as pd
pd.options.mode.chained_assignment = None  # Suppress warnings
import numpy as np
import pickle as pk

from typing import Union

# from typing import List, Union, Dict
# from sklearn.base import BaseEstimator

# ******* Data Visualization Libraries ****************************

import matplotlib.pyplot as plt

import seaborn as sns

from bokeh.io import output_notebook
output_notebook()
from bokeh.plotting import show,figure
TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
]

# ******** Data science / Machine learning Libraries ***************
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

tf.get_logger().setLevel('ERROR')

from tensorflow import keras
from tensorflow.keras.models import save_model


class TrainCNN:
    """
    Train Convolutional Neural Networks model

    Parameters:
    -----------
    X1_train : array-like
        Row-StandardScaled input spectra for training.
    X1_val : array-like
        Row-StandardScaled input spectra for validation.
    X1_test : array-like
        Row-StandardScaled input spectra for testing.
    X2_train : array-like
        Col-StandardScaled Mix Max of all rows of input spectra for training.
    X2_val : array-like
        Col-StandardScaled Mix Max of all rows of input spectra for validation.
    X2_test : array-like
        Col-StandardScaled Mix Max of all rows of input spectra for testing.
    y1_train : array-like
        Col-StandardScaled target feature 1 for training.
    y1_val : array-like
        Col-StandardScaled target feature 1 for validation.
    y1_test : array-like
        Col-StandardScaled target feature 1 for testing.
    y2_train : array-like
        Col-StandardScaled target feature 2 for training.
    y2_val : array-like
        Col-StandardScaled target feature 2 for validation.
    y2_test : array-like
        Col-StandardScaled target feature 2 for testing.
    y3_train : array-like
        Col-StandardScaled target feature 3 for training.
    y3_val : array-like
        Col-StandardScaled target feature 3 for validation.
    y3_test : array-like
        Col-StandardScaled target feature 3 for testing.
    y4_train : array-like
        Col-StandardScaled target feature 4 for training.
    y4_val : array-like
        Col-StandardScaled target feature 4 for validation.
    y4_test : array-like
        Col-StandardScaled target feature 4 for testing.
    """
    def __init__(self,
                 X1_train: Union[np.ndarray, list],
                 X1_val: Union[np.ndarray, list],
                 X1_test: Union[np.ndarray, list],
                 X2_train: Union[np.ndarray, list],
                 X2_val: Union[np.ndarray, list],
                 X2_test: Union[np.ndarray, list],
                 y1_train: Union[np.ndarray, list],
                 y1_val: Union[np.ndarray, list],
                 y1_test: Union[np.ndarray, list],
                 y2_train: Union[np.ndarray, list],
                 y2_val: Union[np.ndarray, list],
                 y2_test: Union[np.ndarray, list],
                 y3_train: Union[np.ndarray, list],
                 y3_val: Union[np.ndarray, list],
                 y3_test: Union[np.ndarray, list],
                 y4_train: Union[np.ndarray, list],
                 y4_val: Union[np.ndarray, list],
                 y4_test: Union[np.ndarray, list]
                 ):

        # train, val, test sets for main features (104 wavelengths)
        self.X1_train, self.X1_val, self.X1_test = X1_train, X1_val, X1_test

        # train, val, test sets for input 2 (Min and Max 2 features)
        self.X2_train, self.X2_val, self.X2_test = X2_train, X2_val, X2_test

        # train, val, test sets for target features
        self.y1_train, self.y1_val, self.y1_test = y1_train, y1_val, y1_test
        self.y2_train, self.y2_val, self.y2_test = y2_train, y2_val, y2_test
        self.y3_train, self.y3_val, self.y3_test = y3_train, y3_val, y3_test
        self.y4_train, self.y4_val, self.y4_test = y4_train, y4_val, y4_test

    def build_model(self,
                    hyperparameters # dic
                    ):
        """
        Build a CNN model with a certain number of blocks and layers using for loops.

        Args:
            hyperparameters (dict): A dictionary containing hyperparameters for configuring the model.
                'Conv__num_blocks' (int): Number of blocks in the CNN model.
                'Conv__num_layers_per_block' (int): Number of layers in each convolutional block.
                'Conv__num_filters' (list): List of integers representing the number of filters in each layer.
                'Conv__kernel_size' (int): Size of the convolutional kernel.
                'Conv__MaxPooling1D' (bool): Whether to include MaxPooling1D layers after each block.

                'FC1__num_blocks' (int): Number of blocks in the fully connected (FC1) part of the model.
                'FC1_num_layers_per_block' (int): Number of layers in each FC1 block.
                'FC1__units' (list): List of integers representing the number of units in each layer.
                'FC1__dropout' (float): Dropout rate for FC1 layers.

                'FC2__num_blocks' (int): Number of blocks in the second fully connected (FC2) part of the model.
                'FC2_num_layers_per_block' (int): Number of layers in each FC2 block.
                'FC2__units' (list): List of integers representing the number of units in each layer.
                'FC2__dropout' (float): Dropout rate for FC2 layers.

                'learning_rate' (float): Learning rate for the model.

        Returns:
            tf.keras.Model: The built CNN model.
        """

        Conv__num_blocks = hyperparameters['Conv__num_blocks']
        Conv__num_layers_per_block = hyperparameters['Conv__num_layers_per_block']
        Conv__num_filters = hyperparameters['Conv__num_filters']
        Conv__kernel_size = hyperparameters['Conv__kernel_size']
        Conv__MaxPooling1D = hyperparameters['Conv__MaxPooling1D']

        FC1__num_blocks = hyperparameters['FC1__num_blocks']
        FC1_num_layers_per_block = hyperparameters['FC1_num_layers_per_block']
        FC1__units = hyperparameters['FC1__units']
        FC1__dropout = hyperparameters['FC1__dropout']

        FC2__num_blocks = hyperparameters['FC2__num_blocks']
        FC2_num_layers_per_block = hyperparameters['FC2_num_layers_per_block']
        FC2__units = hyperparameters['FC2__units']
        FC2__dropout = hyperparameters['FC2__dropout']

        self.learning_rate = hyperparameters['learning_rate']

        # Define the input layer
        input_layer_1 = tf.keras.layers.Input(shape=(104, 1))
        input_layer_2 = tf.keras.layers.Input(shape=(2,))

        # Start building the model using the input layer
        x = input_layer_1

        # Build the specified number of blocks using for loops
        for block in range(Conv__num_blocks):
            # Build the specified number of layers in each block using for loops
            for layer in range(Conv__num_layers_per_block):
                x = Conv1D(filters=Conv__num_filters * ((block + layer + 1) * 2) ** 2,
                           kernel_size=Conv__kernel_size,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal',
                           name='Conv__B' + str(block + 1) + '_L' + str(layer + 1)
                           )(x)

            # Add a MaxPooling layer at the end of each block
            x = MaxPooling1D(pool_size=(Conv__MaxPooling1D),
                             name='MaxPool1D__B' + str(block + 1) + '_L' + str(layer + 1))(x)

        # Flatten the output of the last block
        x = Flatten()(x)

        for block in range(FC1__num_blocks):
            for layer in range(FC1_num_layers_per_block):
                x = Dense(FC1__units * ((block + layer + 1) * 2) ** 2,
                          activation='relu',
                          name='FC1__B' + str(block + 1) + '_L' + str(layer + 1))(x)

            x = Dropout(FC1__dropout,
                        name='FC1__Dropout__B' + str(block + 1) + '_L' + str(layer + 1))(x)

        # Concatenate the outputs from the convolutional layers and dense layer
        x = tf.keras.layers.concatenate([x, input_layer_2],
                                        name='Concatenated_Layer')

        # Add a dense layer for classification
        for block in range(FC2__num_blocks):
            for layer in range(FC2_num_layers_per_block):
                x = Dense(FC2__units * ((block + layer + 1) * 2) ** 2,
                          activation='relu',
                          name='FC2__B' + str(block + 1) + '_L' + str(layer + 1))(x)

            x = Dropout(FC2__dropout,
                        name='FC2__Dropout__B' + str(block + 1) + '_L' + str(layer + 1))(x)

        ######### 3rd FC Block: gravity  ##############################

        out__gravity = Dense(1,
                             activation='linear',
                             # kernel_initializer = 'he_normal',
                             name='gravity')(x)

        ######### 3rd FC Block: c_o_ratio  ##############################
        out__c_o_ratio = Dense(1,
                               activation='linear',
                               # kernel_initializer = 'he_normal',
                               name='c_o_ratio')(x)

        ######### 3rd FC Block: metallicity  ##############################
        out__metallicity = Dense(1,
                                 activation='linear',
                                 # kernel_initializer = 'he_normal',
                                 name='metallicity')(x)

        ######### 3rd FC Block: temperature  ##############################
        out__temperature = Dense(1,
                                 activation='linear',
                                 name='temperature')(x)

        ######### OUTPUT   ################################################
        # Create the model with two inputs and two outputs
        model = tf.keras.Model(inputs=[input_layer_1, input_layer_2],
                               outputs=[out__gravity, out__c_o_ratio, out__metallicity, out__temperature])

        self.model = model

        print(model.summary())

    def fit_cnn_model(self,
                      budget=3):

        model = self.model
        # Compile the model with an optimizer, loss function, and metrics
        model.compile(loss='huber_loss',
                           optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                           metrics=['mae'])

        early_stop = EarlyStopping(monitor='loss', min_delta=4e-4, patience=50, mode='auto', \
                                   restore_best_weights=True)

        # YOU CAN ADD FUNCTION HERE TO ADD NOISE
        history = self.model.fit(x=[self.X1_train, self.X2_train],
                                 y=[self.y1_train, self.y2_train, self.y3_train, self.y4_train],
                                 # self.x_train, self.y_train,
                                 batch_size=32,  # config['batch_size'], # self.batch_size,
                                 validation_data=(
                                 [self.X1_val, self.X2_val], [self.y1_val, self.y2_val, self.y3_val, self.y4_val]),
                                 # validation_split=0.2,
                                 epochs=int(budget),
                                 verbose=1,
                                 callbacks=[early_stop],
                                 )
        self.model = model
        self.history = history
        return history, model
#         train_score = model.evaluate(x = [self.X1_train, self.X2_train],
#                                      y = [self.y1_train, self.y2_train, self.y3_train, self.y4_train],
#                                      verbose=0)
#         val_score   = model.evaluate(x = [self.X1_val, self.X2_val],
#                                      y = [self.y1_val, self.y2_val, self.y3_val, self.y4_val],
#                                      verbose=0)
#         test_score  = model.evaluate(x = [self.X1_test, self.X2_test],
#                                      y = [self.y1_test, self.y2_test, self.y3_test, self.y4_test],
#                                      verbose=0)


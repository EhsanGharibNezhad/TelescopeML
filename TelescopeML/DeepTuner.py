import scipy.signal.signaltools


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


scipy.signal.signaltools._centered = _centered

import os

# os.environ['NUMEXPR_MAX_THREADS'] = '42'
# os.environ['NUMEXPR_NUM_THREADS'] = '40'
# import numexpr as ne

ip_address = '127.0.0.6'

from TelescopeML.DataMaster import *
from TelescopeML.Predictor import *
from TelescopeML.StatVisAnalyzer import *
from TelescopeML.IO_utils import load_or_dump_trained_model_CNN

import logging

logging.basicConfig(level=logging.DEBUG)

# Libraries for BOHB Package
import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
# from hpbandster.examples.commons import MyWorker

# import logging
# logging.basicConfig(level=logging.ERROR) # Change the level to INFO or higher
# import argparse

# import hpbandster.core.nameserver as hpns
# import hpbandster.core.result as hpres

# from hpbandster.optimizers import BOHB as BOHB
# from hpbandster.examples.commons import MyWorker

import numpy
import time

import ConfigSpace as CS
from hpbandster.core.worker import Worker

# Import functions from other modules ============================
# from io_funs import LoadSave

# Import python libraries ========================================
import numpy as np
import matplotlib.pyplot as plt

# Import BOHB libraries ========================================


import ConfigSpace.hyperparameters as CSH

# import logging
# logging.basicConfig(level=logging.DEBUG)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv1D, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Input, Conv1D, Concatenate, Dense, MaxPooling1D, Dropout
from tensorflow.keras.models import Model

# mlp for multi-output regression
from numpy import mean
from numpy import std
from keras.models import Sequential
from keras.layers import Dense

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging

logging.basicConfig(level=logging.DEBUG)


class KerasWorker(Worker):
    def __init__(self,
                 X1_train, X1_val, X1_test,  # Row-StandardScaled input spectra
                 X2_train, X2_val, X2_test,  # Col-StandardScaled Mix Max of all rows of input spetra
                 y1_train, y1_val, y1_test,  # Col-StandardScaled target feature 1
                 y2_train, y2_val, y2_test,  # Col-StandardScaled target feature 2
                 y3_train, y3_val, y3_test,  # Col-StandardScaled target feature 3
                 y4_train, y4_val, y4_test,  # Col-StandardScaled target feature 4
                 *args, sleep_interval=0, **kwargs):

        super().__init__(**kwargs)
        self.sleep_interval = sleep_interval

        # self.batch_size = 2**9

        # train, val, test sets for input 1 (main 104 spectral features)
        self.X1_train, self.X1_val, self.X1_test = X1_train, X1_val, X1_test

        # train, val, test sets for input 2 (Min and Max 2 features)
        self.X2_train, self.X2_val, self.X2_test = X2_train, X2_val, X2_test

        # train, val, test sets for target features
        self.y1_train, self.y1_val, self.y1_test = y1_train, y1_val, y1_test
        self.y2_train, self.y2_val, self.y2_test = y2_train, y2_val, y2_test
        self.y3_train, self.y3_val, self.y3_test = y3_train, y3_val, y3_test
        self.y4_train, self.y4_val, self.y4_test = y4_train, y4_val, y4_test

        # self.input_shape = (104,1)
        # print(np.shape(self.x_train), np.shape(self.y_train))

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Convolution Neural Networks to be optimized by BOHB package.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        Conv__filters = config['Conv__filters']
        Conv__kernel_size = config['Conv__kernel_size']
        Conv__MaxPooling1D = config['Conv__MaxPooling1D']
        Conv__NumberLayers = config['Conv__NumberLayers']
        Conv__NumberBlocks = config['Conv__NumberBlocks']

        FC1__units = config['FC1__units']
        FC1__dropout = config['FC1__dropout']
        FC1__NumberLayers = config['FC1__NumberLayers']

        FC2__units = config['FC2__units']
        FC2__NumberLayers = config['FC2__NumberLayers']
        FC2__dropout = config['FC2__dropout']
        FC2__NumberBlocks = config['FC2__NumberBlocks']

        # FC3__units_temperature = config['FC3__units_temperature']
        # FC3__units_c_o_ratio = config['FC3__units_c_o_ratio']
        # FC3__units_gravity = config['FC3__units_gravity']
        # FC3__units_metallicity = config['FC3__units_metallicity']

        lr = config['lr']

        # for key, value in zip(config.keys(), config.values()):
        #     print(f'{key}: {value}')

        ######### Shape of the inputs
        input_1 = tf.keras.layers.Input(shape=(104, 1))
        input_2 = tf.keras.layers.Input(shape=(2,))

        ######### Conv Blocks  ####################################
        model = input_1
        for b in range(0, Conv__NumberBlocks):
            for l in range(0, Conv__NumberLayers):
                model = Conv1D(filters=Conv__filters * (b + l + 1) ** 2,
                               kernel_size=Conv__kernel_size,
                               strides=1,
                               padding='same',
                               activation='relu',
                               kernel_initializer='he_normal',
                               # kernel_regularizer=tf.keras.regularizers.l2(Conv__regularizer),
                               name='Conv__B' + str(b + 1) + '_L' + str(l + 1))(
                    model)  # (model if l!= 0 and b!= 0 else input_1)

            model = MaxPooling1D(pool_size=(Conv__MaxPooling1D),
                                 name='Conv__B' + str(b + 1) + '__MaxPooling1D')(model)

        ######### Flatten Layer   ####################################
        model = Flatten()(model)

        ######### FC Layer before the Concatenation   ################
        for l in range(FC1__NumberLayers):
            model = Dense(FC1__units * (l + 1) ** 2,
                          activation='relu',
                          kernel_initializer='he_normal',
                          # kernel_regularizer=tf.keras.regularizers.l2(Conv__regularizer),
                          name='FC1__B1_L' + str(l + 1))(model)

            model = Dropout(FC2__dropout,
                            name='FC1__B1_L' + str(l + 1) + '__Dropout')(model)

        ######### Concatenation Layer  ###############################
        # Concatenate the outputs from the convolutional layers and dense layer
        model = tf.keras.layers.concatenate([model, input_2],
                                            name='Concatenated_Layer')

        ######### FC Block  ####################################
        for b in range(0, FC2__NumberBlocks):
            for l in range(0, FC2__NumberLayers):
                model = Dense(FC2__units * (b + l + 1) ** 2,
                              activation='relu',
                              kernel_initializer='he_normal',
                              # kernel_regularizer=tf.keras.regularizers.l2(Conv__regularizer),
                              name='FC2__B' + str(b + 1) + '_L' + str(l + 1))(
                    model)  # (model if l!= 0 and b!= 0 else input_1)

            model = Dropout(FC2__dropout,
                            name='FC2__B' + str(b + 1) + '_L' + str(l + 1) + '__Dropout')(model)

        ######### 3rd FC Block: gravity  #############################

        out__gravity = Dense(1,
                             activation='linear',
                             # kernel_initializer = 'he_normal',
                             name='output__gravity')(model)

        ######### 3rd FC Block: c_o_ratio  ##############################
        out__c_o_ratio = Dense(1,
                               activation='linear',
                               # kernel_initializer = 'he_normal',
                               # kernel_regularizer=tf.keras.regularizers.l2(0.003/2),
                               name='output__c_o_ratio')(model)

        ######### 3rd FC Block: metallicity  ##############################

        out__metallicity = Dense(1,
                                 activation='linear',
                                 # kernel_initializer = 'he_normal',
                                 name='output__metallicity')(model)

        ######### 3rd FC Block: temperature  ##############################
        out__temperature = Dense(1,
                                 activation='linear',
                                 name='output__temperature')(model)

        ######### OUTPUT   ################################################
        # Create the model with two inputs and two outputs
        model = tf.keras.Model(inputs=[input_1, input_2],
                               outputs=[out__gravity, out__c_o_ratio, out__metallicity, out__temperature])

        # print model summary
        # print(model.summary())

        # Compile the model with an optimizer, loss function, and metrics
        model.compile(loss='huber_loss',
                      optimizer=keras.optimizers.Adam(learning_rate=lr),
                      metrics=['mae'])

        early_stop = EarlyStopping(monitor='loss', min_delta=4e-4, patience=50, mode='auto', \
                                   restore_best_weights=True)

        # YOU CAN ADD FUNCTION HERE TO ADD NOISE
        history = model.fit(x=[self.X1_train, self.X2_train],
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

        train_score = model.evaluate(x=[self.X1_train, self.X2_train],
                                     y=[self.y1_train, self.y2_train, self.y3_train, self.y4_train],
                                     verbose=0)
        val_score = model.evaluate(x=[self.X1_val, self.X2_val],
                                   y=[self.y1_val, self.y2_val, self.y3_val, self.y4_val],
                                   verbose=0)
        test_score = model.evaluate(x=[self.X1_test, self.X2_test],
                                    y=[self.y1_test, self.y2_test, self.y3_test, self.y4_test],
                                    verbose=0)

        # print(train_score, val_score, test_score)
        #             #import IPython; IPython.embed()
        return ({
            'loss': val_score[0],  # remember: HpBandSter always minimizes!
            'info': {'test accuracy': test_score,
                     'train accuracy': train_score,
                     'validation accuracy': val_score,
                     'number of parameters': model.count_params(),
                     },
            # 'model' : model,
            # 'history' : history,

        })

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        # Conv hyperparameters
        Conv__filters = CategoricalHyperparameter(name='Conv__filters', choices=[16, 32,
                                                                                 64])  # NOTE: Apply the same categorical method for other unit and
        Conv__kernel_size = UniformIntegerHyperparameter(name='Conv__kernel_size', lower=1, upper=8, default_value=3,
                                                         log=False)  # ok
        Conv__MaxPooling1D = UniformIntegerHyperparameter(name='Conv__MaxPooling1D', lower=1, upper=8, default_value=2,
                                                          log=False)  # ok
        Conv__NumberLayers = UniformIntegerHyperparameter(name='Conv__NumberLayers', lower=1, upper=4, default_value=3,
                                                          log=False)  # ok
        Conv__NumberBlocks = UniformIntegerHyperparameter(name='Conv__NumberBlocks', lower=1, upper=4, default_value=1,
                                                          log=False)  # ok

        FC1__units = CategoricalHyperparameter(name='FC1__units', choices=[8, 16, 32, 64, 128, 256])  # same
        # FC1__NumberBlocks = UniformIntegerHyperparameter(name='FC1__NumberBlocks', lower=1, upper=4, default_value=1,  log=False) ## DELETE,
        FC1__NumberLayers = UniformIntegerHyperparameter(name='FC1__NumberLayers', lower=1, upper=4, default_value=1,
                                                         log=False)  ### DELETE
        FC1__dropout = UniformFloatHyperparameter(name='FC1__dropout', lower=0.001, upper=0.4, default_value=0.02,
                                                  log=True)

        # FC hyperparameters
        FC2__units = CategoricalHyperparameter(name='FC2__units', choices=[8, 16, 32, 64, 128,
                                                                           256])  # NOTE: Apply the same categorical method for other unit and
        FC2__NumberLayers = UniformIntegerHyperparameter(name='FC2__NumberLayers', lower=1, upper=4, default_value=2,
                                                         log=False)
        FC2__NumberBlocks = UniformIntegerHyperparameter(name='FC2__NumberBlocks', lower=1, upper=4, default_value=1,
                                                         log=False)  # DELETE - No blocks for FC
        FC2__dropout = UniformFloatHyperparameter(name='FC2__dropout', lower=0.001, upper=0.4, default_value=0.02,
                                                  log=True)

        # FC3__units_temperature = CategoricalHyperparameter(name='FC3__units_temperature', choices=[8, 16, 32 , 64, 128, 256]) # the same
        # FC3__units_metallicity = CategoricalHyperparameter(name='FC3__units_metallicity', choices=[8, 16, 32 , 64, 128, 256]) # the same
        # FC3__units_c_o_ratio = CategoricalHyperparameter(name='FC3__units_c_o_ratio', choices=[8, 16, 32 , 64, 128, 256]) # the same
        # FC3__units_gravity = CategoricalHyperparameter(name='FC3__units_gravity', choices=[8, 16, 32 , 64, 128, 256]) # same

        # Other hyperparameters
        lr = UniformFloatHyperparameter(name='lr', lower=1e-5, upper=1e-2, default_value=1e-4, log=True)

        cs.add_hyperparameters([
            Conv__filters,
            Conv__kernel_size,
            Conv__MaxPooling1D,
            Conv__NumberLayers,
            Conv__NumberBlocks,
            #
            FC1__NumberLayers,
            FC1__units,
            FC1__dropout,
            #
            FC2__units,
            FC2__NumberLayers,
            FC2__NumberBlocks,
            FC2__dropout,
            #
            # FC3__units_temperature,
            # FC3__units_c_o_ratio,
            # FC3__units_gravity,
            # FC3__units_metallicity,
            #
            lr,
        ])

        return cs


# ==============================================================


# Read the `TelescopeML_reference_data` path

import os  # to check the path
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    CategoricalHyperparameter

__reference_data_path__ = os.getenv("TelescopeML_reference_data")
__reference_data_path__

# Step 1: Load the dataset
train_BD = pd.read_csv(os.path.join(__reference_data_path__,
                                    'training_datasets',
                                    'browndwarf_R100_v4_newWL_v2.csv.bz2'), compression='bz2')

wl_synthetic = pd.read_csv(os.path.join(__reference_data_path__,
                                        'training_datasets',
                                        'wl.csv'))

## Prepare feature variables (X) and targets (y)
# to assure we are only training the module with the native non-augmented BD training dataset
train_BD = train_BD[train_BD['is_augmented'].isin(['no'])]

target_features = ['gravity', 'temperature', 'c_o_ratio', 'metallicity']
training_features_labels = [item for item in train_BD.columns.to_list() if
                            item not in target_features + ['is_augmented']]

# Training feature variables
X = train_BD.drop(
    columns=['gravity',
             'temperature',
             'c_o_ratio',
             'metallicity',
             'is_augmented'])  # .astype(np.float32)

# Target/Output feature variables
y = train_BD[['gravity', 'c_o_ratio', 'metallicity', 'temperature', ]]  # .astype(np.float32)

# Log-Transform
y.loc[:, 'temperature'] = np.log10(y['temperature'])
# df['temperature'] = df['temperature'].apply(lambda x: np.log10(x))

# Create an instance of TrainCNNRegression
data_processor = DataProcessor(
    #                              trained_model = None,
    #                              trained_model_history = None,
    feature_values=X.to_numpy(),
    feature_names=X.columns,
    target_values=y.to_numpy(),
    target_name=['gravity', 'c_o_ratio', 'metallicity', 'temperature'],
    is_tuned='yes',
    param_grid=None,
    spectral_resolution=100,
    is_feature_improved='no',
    is_augmented='no',
    ml_model=None,
    ml_model_str='CNN',
)

# Split the dataset into train and test sets
data_processor.split_train_validation_test(test_size=0.1,
                                           val_size=0.1,
                                           random_state_=42, )

# Scale the X features using MinMax Scaler
data_processor.standardize_X_row_wise(output_indicator='Trained_StandardScaler_X_RowWise')

# Standardize the y features using Standard Scaler
data_processor.standardize_y_column_wise(output_indicator='Trained_StandardScaler_y_ColWise')

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

df_MinMax_train.rename(columns={0: 'min', 1: 'max'}, inplace=True)
df_MinMax_val.rename(columns={0: 'min', 1: 'max'}, inplace=True)
df_MinMax_test.rename(columns={0: 'min', 1: 'max'}, inplace=True)

data_processor.standardize_X_column_wise(
    output_indicator='Trained_StandardScaler_X_ColWise_MinMax',
    X_train=df_MinMax_train.to_numpy(),
    X_val=df_MinMax_val.to_numpy(),
    X_test=df_MinMax_test.to_numpy(),
)

# data_processor.plot_boxplot_scaled_features(scaled_feature= data_processor.X_test_standardized_columnwise,
#                                               xticks_list = ['','Min','Max'],
#                                               title = 'Scaled Min Max Features - ColumnWise',
#                                               fig_size=(4, 3),
#                                                 )

train_cnn_regression = data_processor

# =========================================================


import logging

logging.basicConfig(level=logging.DEBUG)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
# from hpbandster.examples.commons import MyWorker


import argparse
import sys

# Check if running as a Jupyter kernel
if "ipykernel_launcher" in sys.argv[0]:
    args = argparse.Namespace(
        min_budget=15,
        max_budget=30,
        n_iterations=50000,
        n_workers=40,
        worker=False
    )
else:
    # Create the optimization object with the desired queue size
    parser = argparse.ArgumentParser(description='Example 3 - Local and Parallel Execution.')
    parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=20)
    parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=50)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer',
                        default=50000)
    parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=40)
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')

    args = parser.parse_args()

# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.

NS = hpns.NameServer(run_id='example1', host=ip_address, port=None)
NS.start()

# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
w = KerasWorker(
    # input dataset: StandardScaled instances
    X1_train=train_cnn_regression.X_train_standardized_rowwise,
    X1_val=train_cnn_regression.X_val_standardized_rowwise,
    X1_test=train_cnn_regression.X_test_standardized_rowwise,

    # input dataset: Min Max of each instance
    X2_train=train_cnn_regression.X_train_standardized_columnwise,
    X2_val=train_cnn_regression.X_val_standardized_columnwise,
    X2_test=train_cnn_regression.X_test_standardized_columnwise,

    # 1st target
    y1_train=train_cnn_regression.y_train_standardized_columnwise[:, 0],
    y1_val=train_cnn_regression.y_val_standardized_columnwise[:, 0],
    y1_test=train_cnn_regression.y_test_standardized_columnwise[:, 0],

    # 2nd target
    y2_train=train_cnn_regression.y_train_standardized_columnwise[:, 1],
    y2_val=train_cnn_regression.y_val_standardized_columnwise[:, 1],
    y2_test=train_cnn_regression.y_test_standardized_columnwise[:, 1],

    # 3rd target
    y3_train=train_cnn_regression.y_train_standardized_columnwise[:, 2],
    y3_val=train_cnn_regression.y_val_standardized_columnwise[:, 2],
    y3_test=train_cnn_regression.y_test_standardized_columnwise[:, 2],

    # 4th target
    y4_train=train_cnn_regression.y_train_standardized_columnwise[:, 3],
    y4_val=train_cnn_regression.y_val_standardized_columnwise[:, 3],
    y4_test=train_cnn_regression.y_test_standardized_columnwise[:, 3],
    nameserver=ip_address, run_id='example1')
w.run(background=True)

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run BOHB, but that is not essential.
# The run method will return the `Result` that contains all runs performed.
result_logger = hpres.json_result_logger(directory='out9/', overwrite=True)

bohb = BOHB(configspace=w.get_configspace(),
            run_id='example1', nameserver=ip_address,
            min_budget=args.min_budget, max_budget=args.max_budget,
            result_logger=result_logger,
            )
res = bohb.run(n_iterations=args.n_iterations)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds informations about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.' % (
            sum([r.budget for r in res.get_all_runs()]) / args.max_budget))
print('The run took  %.1f seconds to complete.' % (
            all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))

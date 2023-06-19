import os
os.environ['NUMEXPR_MAX_THREADS'] = '12'
os.environ['NUMEXPR_NUM_THREADS'] = '10'
import numexpr as ne

ip_address = '127.0.0.1'


#import logging
#logging.basicConfig(level=logging.INFO)


import sys
sys.path.insert(0, '../../codes/')

from predict_observational_dataset_v2 import ProcessObservationalDataset
from check_results_regression_2 import *
from bohb_optimizer_2 import *
# from train_ml_regression_2 import *
from train_cnn_regression_3 import *
from search_space import *
# from bohb_worker import *

# Libraries for BOHB Package
import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
# from hpbandster.examples.commons import MyWorker

#import logging
#logging.basicConfig(level=logging.ERROR) # Change the level to INFO or higher
#import argparse

#import hpbandster.core.nameserver as hpns
#import hpbandster.core.result as hpres

#from hpbandster.optimizers import BOHB as BOHB
# from hpbandster.examples.commons import MyWorker

import numpy
import time

import ConfigSpace as CS
from hpbandster.core.worker import Worker
from sklearn.preprocessing import PowerTransformer




# Import functions from other modules ============================
from io_funs import LoadSave

# Import python libraries ========================================
import numpy as np
import matplotlib.pyplot as plt


# Import python libraries ========================================
import numpy as np
import matplotlib.pyplot as plt

# Import BOHB libraries ========================================


import ConfigSpace.hyperparameters as CSH

#import logging
#logging.basicConfig(level=logging.DEBUG)


import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv1D, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Input, Conv1D, Concatenate, Dense, MaxPooling1D
from tensorflow.keras.models import Model

# mlp for multi-output regression
from numpy import mean
from numpy import std
from keras.models import Sequential
from keras.layers import Dense

import warnings
import logging
  
  
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter
# Set the logging level to CRITICAL
#logging.basicConfig(level=logging.CRITICAL)
# Libraries for BOHB Package 
#import logging
#logging.basicConfig(level=logging.WARNING)

#import argparse

#import hpbandster.core.nameserver as hpns
#import hpbandster.core.result as hpres


# ===============================================================================
# ==================                                           ==================
# ==================            BOHB Optimizer                 ==================
# ==================                                           ==================
# ===============================================================================  




from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter
# Set the logging level to CRITICAL
logging.basicConfig(level=logging.CRITICAL)
    
class KerasWorker(Worker):
    def __init__(self, 
                   X1_train, X1_val, X1_test, # Row-StandardScaled input spectra
                   X2_train, X2_val, X2_test, # Col-StandardScaled Mix Max of all rows of input spetra
                   y1_train, y1_val, y1_test, # Col-StandardScaled target feature 1
                   y2_train, y2_val, y2_test, # Col-StandardScaled target feature 2
                   y3_train, y3_val, y3_test, # Col-StandardScaled target feature 3
                   y4_train, y4_val, y4_test, # Col-StandardScaled target feature 4                 
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
        
        FC__units = config['FC__units']
        FC__units_temperature = config['FC__units_temperature']
        FC__units_c_o_ratio = config['FC__units_c_o_ratio']
        FC__units_gravity = config['FC__units_gravity']
        FC__units_metallicity = config['FC__units_metallicity']
        FC__NumberBlocks = config['FC__NumberBlocks']
        FC__NumberLayers = config['FC__NumberLayers']
        
        FC_in_Conv__units = config['FC_in_Conv__units']
        FC_in_Conv__NumberLayers = config['FC_in_Conv__NumberLayers']
        FC_in_Conv__NumberBlocks = config['FC_in_Conv__NumberBlocks']
        
        
        FC__dropout = config['FC__dropout']
        FC_out_dropout = config['FC_out_dropout']
        

        lr = config['lr']
        LeakyReLU_alpha = config['LeakyReLU_alpha']
        kernel_initializer_list = config['kernel_initializer_list']   
        

        ######### Shape of the inputs
        input_1 = tf.keras.layers.Input(shape=(104, 1))
        input_2 = tf.keras.layers.Input(shape=(2,))

        
        ######### Conv Blocks  ####################################
        model = input_1
        for b in range(0, Conv__NumberBlocks):
            for l in range(0, Conv__NumberLayers):
                model = Conv1D(filters = Conv__filters*(2)**b, 
                                  kernel_size = Conv__kernel_size, 
                                  strides = 1, 
                                  padding ='same', 
                                  activation = LeakyReLU(alpha=LeakyReLU_alpha), 
                                  kernel_initializer = 'he_normal',
                                  # kernel_regularizer=tf.keras.regularizers.l2(Conv__regularizer),
                                  name = 'Conv__B'+str(b+1)+'_L'+str(l+1))(model) #(model if l != 0 and b != 0 else input_1)

            model = MaxPooling1D(pool_size=(Conv__MaxPooling1D),
                                name = 'MaxPooling1D__B'+str(b+1)+'_L'+str(l+1))(model)


        ######### Flatten Layer   ####################################
        model = Flatten()(model)


        ######### FC Layer before the Concatenation   ################
        for b in range(FC_in_Conv__NumberBlocks): ## NOTE: NO NEED FOR A BLOCK, just 1 layer
            for l in range(FC_in_Conv__NumberLayers): ## just 1 LAYER
                model = Dense(FC_in_Conv__units, 
                                   activation = LeakyReLU(alpha=LeakyReLU_alpha),
                                   kernel_initializer = 'he_normal',
                                   # kernel_regularizer=tf.keras.regularizers.l2(Conv__regularizer),
                                   name = 'FC_in_Conv__B'+str(b+1)+'_L'+str(l+1))(model)
                
                model= Dropout(FC__dropout, #### Have different Drop out here!!! 
                                name = 'FC_in_Conv__Dropout__B'+str(b+1)+'_L'+str(l+1))(model)
            
        
        
        ######### Concatenation Layer  ###############################
        # Concatenate the outputs from the convolutional layers and dense layer
        model = tf.keras.layers.concatenate([model, input_2], 
                                                           name='Concatenated_Layer')

        
        ######### FC Block  ####################################
        for b in range(FC__NumberBlocks): # We need 1 Blocks
            for l in range(FC__NumberLayers): # We can have multiple layers
                model = Dense(FC__units, 
                           activation = LeakyReLU(alpha=LeakyReLU_alpha),
                           kernel_initializer = 'he_normal',
                           # kernel_regularizer=tf.keras.regularizers.l2(Conv__regularizer),
                           name = 'FC__B'+str(b+1)+'_L'+str(l+1))(model)

                model= Dropout(FC__dropout, 
                                       name = 'FC__Dropout__B'+str(b+1)+'_L'+str(l+1))(model)
        
        

        ######### 3rd FC Block: gravity  ##############################
    #         FC2 = FC__Drop

        model2 = Dense(FC__units_gravity, 
                        activation = LeakyReLU(alpha=LeakyReLU_alpha),
                        kernel_initializer = 'he_normal',
                        # kernel_regularizer=tf.keras.regularizers.l2(0.003/2),
                        name = 'FC_block3_gravity')(model)
        
        model2= Dropout(FC_out_dropout, 
                               name = 'FC_block3_gravity__Dropout')(model2)
        
        out__gravity = Dense(1, 
                             activation = 'linear',
                             kernel_initializer = 'he_normal',
                             name = 'gravity')(model2)
        

        
        ######### 3rd FC Block: c_o_ratio  ##############################
        model2 = Dense(FC__units_c_o_ratio, 
                        activation = LeakyReLU(alpha=LeakyReLU_alpha),
                        kernel_initializer = 'he_normal',
                        # kernel_regularizer=tf.keras.regularizers.l2(0.003/2),
                        name = 'FC_block3_c_o_ratio')(model)
        
        model2= Dropout(FC_out_dropout, 
                               name = 'FC_block3_c_o_ratio__Dropout')(model2)


        out__c_o_ratio = Dense(1, 
                               activation = 'linear',
                               kernel_initializer = 'he_normal',
                               # kernel_regularizer=tf.keras.regularizers.l2(0.003/2),
                               name='c_o_ratio')(model2)

        
        ######### 3rd FC Block: metallicity  ##############################
        model2 = Dense(FC__units_metallicity, 
                        activation = LeakyReLU(alpha=LeakyReLU_alpha),
                        kernel_initializer = 'he_normal',
                        # kernel_regularizer=tf.keras.regularizers.l2(0.003/2),
                        name = 'FC_block3_metallicity')(model)
        
        model2= Dropout(FC_out_dropout, 
                               name = 'FC_block3_metallicity__Dropout')(model2)

        
        out__metallicity = Dense(1, 
                                 activation = 'linear',
                                 kernel_initializer = 'he_normal',
                                 name='metallicity')(model2)
        
        
        
        ######### 3rd FC Block: temperature  ##############################
        model2 = Dense(FC__units_temperature, 
                        activation = LeakyReLU(alpha=LeakyReLU_alpha),
                        kernel_initializer = 'he_normal',
                        # kernel_regularizer=tf.keras.regularizers.l2(0.003/2),
                        name = 'FC_block3_temperature')(model)
        
        model2= Dropout(FC_out_dropout, 
                               name = 'FC_block3_temperature__Dropout')(model2)
        

        out__temperature = Dense(1, 
                                 activation = 'linear',
                                 name='temperature')(model2)


        ######### OUTPUT   ################################################
        # Create the model with two inputs and two outputs
        model = tf.keras.Model(inputs=[input_1, input_2], 
                               outputs=[out__gravity, out__c_o_ratio, out__metallicity, out__temperature])

        # Compile the model with an optimizer, loss function, and metrics
        model.compile(loss='huber_loss', 
                      optimizer=keras.optimizers.Adam(lr = lr),  
                      metrics=['mae'])
        
        
  
        early_stop = EarlyStopping(monitor='loss', min_delta=4e-4, patience=50, mode='auto', \
                                       restore_best_weights=True)



        # YOU CAN ADD FUNCTION HERE TO ADD NOISE
        history = model.fit(x = [self.X1_train, self.X2_train], 
                            y = [self.y1_train, self.y2_train, self.y3_train, self.y4_train],  #self.x_train, self.y_train,
                          batch_size = 32, #config['batch_size'], # self.batch_size,
                          validation_data=([self.X1_val, self.X2_val], [self.y1_val, self.y2_val, self.y3_val, self.y4_val]),
                          # validation_split=0.2,
                          epochs=int(budget),
                          verbose=1,
                          callbacks=[early_stop],
                 )


        train_score = model.evaluate(x = [self.X1_train, self.X2_train], 
                                     y = [self.y1_train, self.y2_train, self.y3_train, self.y4_train],
                                     verbose=0)
        val_score   = model.evaluate(x = [self.X1_val, self.X2_val], 
                                     y = [self.y1_val, self.y2_val, self.y3_val, self.y4_val],
                                     verbose=0)
        test_score  = model.evaluate(x = [self.X1_test, self.X2_test], 
                                     y = [self.y1_test, self.y2_test, self.y3_test, self.y4_test],
                                     verbose=0)

        print(train_score, val_score, test_score)
#             #import IPython; IPython.embed()
        return ({
                'loss': 1 - val_score[1], # remember: HpBandSter always minimizes!
                'info': {       'test accuracy': test_score[1],
                                        'train accuracy': train_score[1],
                                        'validation accuracy': val_score[1],
                                        'number of parameters': model.count_params(),
                                },
                #'model' : model,
                #'history' : history,

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
        Conv__filters = CategoricalHyperparameter(name='Conv__filters', choices=[4 , 8, 16, 32]) # NOTE: Apply the same categorical method for other unit and 
        Conv__kernel_size = UniformIntegerHyperparameter(name='Conv__kernel_size', lower=1, upper=8, default_value=1,  log=False) # ok
        Conv__MaxPooling1D = UniformIntegerHyperparameter(name='Conv__MaxPooling1D', lower=1, upper=8, default_value=1, log=False) # ok
        Conv__NumberLayers = UniformIntegerHyperparameter(name='Conv__NumberLayers', lower=1, upper=6, default_value=1,  log=False) # ok
        Conv__NumberBlocks =  UniformIntegerHyperparameter(name='Conv__NumberBlocks', lower=1, upper=4, default_value=1,  log=False) # ok

        # FC hyperparameters
        FC__units = CategoricalHyperparameter(name='FC_units', choices=[8, 16, 32 , 64, 128, 256]) # NOTE: Apply the same categorical method for other unit and 

        #FC__units = UniformIntegerHyperparameter(name='FC__units', lower=32, upper=256, default_value=32, log=False) ####NOTE
        FC__units_temperature = UniformIntegerHyperparameter(name='FC__units_temperature', lower=32, upper=256, default_value=32, log=False) # the same
        FC__units_metallicity = UniformIntegerHyperparameter(name='FC__units_metallicity', lower=32, upper=256, default_value=32, log=False) # the same
        FC__units_c_o_ratio = UniformIntegerHyperparameter(name='FC__units_c_o_ratio', lower=32, upper=256, default_value=32, log=False) # the same
        FC__units_gravity = UniformIntegerHyperparameter(name='FC__units_gravity', lower=32, upper=256, default_value=32, log=False) # same

        FC_in_Conv__units = UniformIntegerHyperparameter(name='FC_in_Conv__units', lower=32, upper=256, default_value=32,  log=False) # same
        FC__NumberLayers = UniformIntegerHyperparameter(name='FC__NumberLayers', lower=1, upper=5, default_value=1,  log=False) 
        FC__NumberBlocks = UniformIntegerHyperparameter(name='FC__NumberBlocks', lower=1, upper=5, default_value=1,  log=False) # DELETE - No blocks for FC
        FC__dropout1 = UniformFloatHyperparameter(name='FC__dropout', lower=0.001, upper=0.4, default_value=0.02, log=True)
        FC__dropout2 = UniformFloatHyperparameter(name='FC__dropout', lower=0.001, upper=0.4, default_value=0.02, log=True)
        #FC__dropout3 = UniformFloatHyperparameter(name='FC__dropout', lower=0.001, upper=0.4, default_value=0.02, log=True)
        FC_out_dropout = UniformFloatHyperparameter(name='FC_out_dropout', lower=0.001, upper=0.4, default_value=0.02, log=True)
        
        #FC_in_Conv__NumberBlocks = UniformIntegerHyperparameter(name='FC_in_Conv__NumberBlocks', lower=1, upper=5, default_value=1,  log=False) ## DELETE, 
        #FC_in_Conv__NumberLayers = UniformIntegerHyperparameter(name='FC_in_Conv__NumberLayers', lower=1, upper=5, default_value=1,  log=False) ### DELETE
        
        # Other hyperparameters
        lr = UniformFloatHyperparameter(name='lr', lower=1e-5, upper=1e-2, default_value=1e-4, log=True)
        #LeakyReLU_alpha = UniformFloatHyperparameter(name='LeakyReLU_alpha', lower=0.01, upper=0.3, default_value=0.01, log=True) ## RELU
        #kernel_initializer_list = CategoricalHyperparameter(name='kernel_initializer_list', choices=['he_normal', 'glorot_uniform']) ## CHECK


        # cs = ConfigurationSpace()
        # notes from Hamed:  
        # Conv1Dinp__kernel_size: # this operator 
        # Kenrel size is the same, Stride not 
        # Conv1Dinp__filters: 1 filter, 2*, 4*, .... Optimize this once
        # Stride = 1
        # Optimize size
        # kernel_regularizer = Default
        # MaxPooling1D: 
        # Gaus = keras.layers.GaussianNoise(0.01,)(input_1)
   
        
        cs.add_hyperparameters([
                                Conv__filters,
                                Conv__kernel_size,
                                Conv__MaxPooling1D,
                                Conv__NumberLayers,
                                Conv__NumberBlocks,
            
                                FC__units,
                                FC__units_temperature,
                                FC__units_c_o_ratio,
                                FC__units_gravity,
                                FC__units_metallicity,
                                FC__NumberLayers,
                                FC__NumberBlocks,
                                FC__dropout,
                                FC_out_dropout,
            
                                FC_in_Conv__units,
                                FC_in_Conv__NumberBlocks,
                                FC_in_Conv__NumberLayers,
            
            
                                lr,
                                LeakyReLU_alpha,
                                kernel_initializer_list
                               ]) 
        
        return cs
    
    
    
    

# Step 1: Load the dataset
df=pd.read_csv('../../datasets/browndwarf_R100_v4_newWL_v2.csv.bz2', compression='bz2')
wl = pd.read_csv('../../datasets/wl.csv')


## Prepare feature variables (X) and targets (y)
df = df[df['is_augmented'].isin(['no'])]
X = df.drop(
    columns=['gravity', 
             'temperature', 
             'c_o_ratio', 
             'metallicity', 
             'is_augmented'])#.astype(np.float32)

y = df[['gravity', 'c_o_ratio', 'metallicity', 'temperature', ]]#.astype(np.float32)


# Log-Transform
qt = PowerTransformer(standardize=False)
y[['gravity', 'c_o_ratio', 'metallicity', 'temperature']] = qt.fit_transform(y)

### Create two new features: X_min, X_max
X['min'] = [X.iloc[i].min() for i in range(len(X))]
X['max'] = [X.iloc[i].max() for i in range(len(X))]


# Step 2: Train CNN model
cnn_model = TrainCNNRegression (
                        trained_model  = None, 
                        feature_values = X, 
                        feature_names  = X.columns,
                        target_values  = y.to_numpy(), 
                        target_name    = ['gravity', 'c_o_ratio', 'metallicity', 'temperature'],
                        is_tuned         = 'yes', # 'yes' if you want to tune the parameters
                        param_grid     = None, # hyperparameters to tune, 
                        spectral_resolution = 100,
                        is_feature_improved = 'no',
                        n_jobs = 10,  # number of processors to use
                        cv     = 5,   # Determines the cross-validation splitting strategy
                        is_augmented =  'no',  # select 'yes' if you are using augmented dataset 
                        ml_model     =   None, 
                        ml_model_str =  'CNN',
            )



# split the brown dwarf dataset into train and test sets
cnn_model.split_train_validation_test()


cnn_model.X2_train = cnn_model.X_train[['min','max']]
cnn_model.X2_val = cnn_model.X_val[['min','max']]
cnn_model.X2_test = cnn_model.X_test[['min','max']] 


cnn_model.X2_train, cnn_model.X2_val, cnn_model.X2_test = cnn_model.MinMax_col_X_3(X_train = cnn_model.X2_train,
                                                                                           X_val =  cnn_model.X2_val,
                                                                                           X_test =  cnn_model.X2_test
                                                                                           )



# Feature scaling to normalize the range of independent variables or features of data.

cnn_model.X_train, cnn_model.X_val, cnn_model.X_test =  cnn_model.MinMax_row_X_3(X_train = cnn_model.X_train.drop(columns=['min','max']),
                                 X_val =  cnn_model.X_val.drop(columns=['min','max']),
                                 X_test =  cnn_model.X_test.drop(columns=['min','max']))





cnn_model.y_train, cnn_model.y_val, cnn_model.y_test =  cnn_model.MinMax_col_y_3(y_train = cnn_model.y_train,
                                 y_val =  cnn_model.y_val,
                                 y_test =  cnn_model.y_test)

import argparse
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
# from hpbandster.examples.commons import MyWorker



# Create the optimization object with the desired queue size


parser = argparse.ArgumentParser(description='Example 3 - Local and Parallel Execution.')
parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=10)
parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=40)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=20) #1000
parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default = 10 )
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')

args=parser.parse_args()



if args.worker:
    upper_index = 10_000

    w = KerasWorker( 
                # input dataset: StandardScaled instances  
                X1_train = cnn_model.X_train[:upper_index].reshape(-1, 104, 1),
                X1_val   = cnn_model.X_val[:upper_index].reshape(-1, 104, 1),
                X1_test  = cnn_model.X_test[:upper_index].reshape(-1, 104, 1),

                # input dataset: Min Max of each instance  
                X2_train = cnn_model.X2_train[:upper_index].reshape(-1, 2),
                X2_val = cnn_model.X2_val[:upper_index].reshape(-1, 2),
                X2_test= cnn_model.X2_test[:upper_index].reshape(-1, 2),

                # 1st target
                y1_train = cnn_model.y_train[:upper_index,0].reshape(-1, 1),
                y1_val   = cnn_model.y_val[:upper_index,0].reshape(-1, 1),
                y1_test  = cnn_model.y_test[:upper_index,0].reshape(-1, 1),

                # 2nd target
                y2_train = cnn_model.y_train[:upper_index,1].reshape(-1, 1),
                y2_val   = cnn_model.y_val[:upper_index,1].reshape(-1, 1),
                y2_test  = cnn_model.y_test[:upper_index,1].reshape(-1, 1),

                # 3rd target
                y3_train = cnn_model.y_train[:upper_index,2].reshape(-1, 1),
                y3_val   = cnn_model.y_val[:upper_index,2].reshape(-1, 1),
                y3_test  = cnn_model.y_test[:upper_index,2].reshape(-1, 1),

                # 4th target
                y4_train = cnn_model.y_train[:upper_index,3].reshape(-1, 1),
                y4_val   = cnn_model.y_val[:upper_index,3].reshape(-1, 1),
                y4_test  = cnn_model.y_test[:upper_index,3].reshape(-1, 1),


                sleep_interval = 0.5, nameserver=ip_address,run_id='example3')

    w.run(background=False)
    exit(0)

# Start a nameserver (see example_1)
NS = hpns.NameServer(run_id='example3', host=ip_address, port=None)
NS.start()


# Run an optimizer (see example_2)
result_logger = hpres.json_result_logger(directory='/data2/ehsan_storage/telescopeML/core_codes/bohb_tuning/out15/', overwrite=True)


bohb = BOHB(  configspace = KerasWorker.get_configspace(),
                      run_id = 'example3',
                      min_budget=args.min_budget, 
		      max_budget=args.max_budget,
		     result_logger = result_logger,
               )
res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)


#scores = res.get_all_runs(only_largest_budget=False)
#df = res.get_pandas_dataframe()[0]

#df['config_id']= [ scores[i].config_id for i in range(len(scores)) ]
#df['loss']= [ scores[i].loss for i in range(len(scores))  ]
#df[['test_accuracy', 'train_accuracy', 'validation_accuracy', 'number_parameters']] = [ list(scores[i].info.values()) for i in range(len(scores)) ]
#df['time_stamps']= [ scores[i].time_stamps['finished'] - scores[i].time_stamps['started'] for i in range(len(scores)) ]
#df.sort_values(by='loss', ascending=False)
#df.to_csv('df_output.csv', columns=df.columns.tolist(), index=False)



# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()


# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds informations about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.


#id2config = res.get_id2config_mapping()
#incumbent = res.get_incumbent_id()

#all_runs = res.get_all_runs()

#print('Best found configuration:', id2config[incumbent]['config'])
#print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
#print('A total of %i runs where executed.' % len(res.get_all_runs()))
#print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/args.max_budget))
#print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/args.max_budget))
##print('The run took  %.1f seconds to complete.'%(all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started'])):


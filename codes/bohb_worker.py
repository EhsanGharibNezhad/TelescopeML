# Import functions from other modules ============================
from io_funs import LoadSave

# Import python libraries ========================================
import numpy as np
import matplotlib.pyplot as plt


# Import python libraries ========================================
import numpy as np
import matplotlib.pyplot as plt

# Import BOHB libraries ========================================

try:
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, concatenate, LeakyReLU
    from keras.callbacks import EarlyStopping
    from tensorflow.keras.regularizers import l1, l2, l1_l2


    from keras import backend as K
except:
    raise ImportError("For this example you need to install keras.")

try:
    import torchvision
    import torchvision.transforms as transforms
except:
    raise ImportError("For this example you need to install pytorch-vision.")



import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)


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
# logging.basicConfig(level=logging.CRITICAL)
# Libraries for BOHB Package 
import logging
logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.examples.commons import MyWorker
import logging
logging.basicConfig(level=logging.ERROR) # Change the level to INFO or higher
import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.examples.commons import MyWorker

import numpy
import time

import ConfigSpace as CS
from hpbandster.core.worker import Worker

# ===============================================================================
# ==================                                           ==================
# ==================            BOHB Optimizer                 ==================
# ==================                                           ==================
# ===============================================================================  




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
                model = Conv1D(filters = Conv__filters*(b+l+1)*2, 
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
        for b in range(FC_in_Conv__NumberBlocks):
            for l in range(FC_in_Conv__NumberLayers):
                model = Dense(FC_in_Conv__units*(b+l+1)*4, 
                                   activation = LeakyReLU(alpha=LeakyReLU_alpha),
                                   kernel_initializer = 'he_normal',
                                   # kernel_regularizer=tf.keras.regularizers.l2(Conv__regularizer),
                                   name = 'FC_in_Conv__B'+str(b+1)+'_L'+str(l+1))(model)
                
                model= Dropout(FC__dropout, 
                                name = 'FC_in_Conv__Dropout__B'+str(b+1)+'_L'+str(l+1))(model)
            
        
        
        ######### Concatenation Layer  ###############################
        # Concatenate the outputs from the convolutional layers and dense layer
        model = tf.keras.layers.concatenate([model, input_2], 
                                                           name='Concatenated_Layer')

        
        ######### FC Block  ####################################
        for b in range(FC__NumberBlocks):
            for l in range(FC__NumberLayers):
                model = Dense(FC__units*(b+l+1)*4, 
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
                          # validation_data=([self.X1_val, self.X2_val], [self.y1_val, self.y2_val, self.y3_val, self.y4_val]),
                          validation_split=0.2,
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

        # print(train_score, val_score, test_score)
#             #import IPython; IPython.embed()
        return ({
                'loss': 1-val_score[1], # remember: HpBandSter always minimizes!
                'info': {       'test accuracy': test_score[1],
                                        'train accuracy': train_score[1],
                                        'validation accuracy': val_score[1],
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
        Conv__filters = UniformIntegerHyperparameter(name='Conv__filters', lower=2, upper=32, default_value=16,  log=False)
        Conv__kernel_size = UniformIntegerHyperparameter(name='Conv__kernel_size', lower=1, upper=8, default_value=1,  log=False)
        Conv__MaxPooling1D = UniformIntegerHyperparameter(name='Conv__MaxPooling1D', lower=1, upper=8, default_value=1, log=False)
        Conv__NumberLayers = UniformIntegerHyperparameter(name='Conv__NumberLayers', lower=1, upper=4, default_value=1,  log=False)
        Conv__NumberBlocks =  UniformIntegerHyperparameter(name='Conv__NumberBlocks', lower=1, upper=4, default_value=1,  log=False)

        # FC hyperparameters
        FC__units = UniformIntegerHyperparameter(name='FC__units', lower=32, upper=128, default_value=32, log=False)
        FC__units_temperature = UniformIntegerHyperparameter(name='FC__units_temperature', lower=32, upper=256, default_value=32, log=False)
        FC__units_metallicity = UniformIntegerHyperparameter(name='FC__units_metallicity', lower=32, upper=256, default_value=32, log=False)
        FC__units_c_o_ratio = UniformIntegerHyperparameter(name='FC__units_c_o_ratio', lower=32, upper=256, default_value=32, log=False)
        FC__units_gravity = UniformIntegerHyperparameter(name='FC__units_gravity', lower=32, upper=256, default_value=32, log=False)
        FC_in_Conv__units = UniformIntegerHyperparameter(name='FC_in_Conv__units', lower=32, upper=128, default_value=32,  log=False)
        FC__NumberLayers = UniformIntegerHyperparameter(name='FC__NumberLayers', lower=1, upper=5, default_value=1,  log=False)
        FC__NumberBlocks = UniformIntegerHyperparameter(name='FC__NumberBlocks', lower=1, upper=5, default_value=1,  log=False)
        FC__dropout = UniformFloatHyperparameter(name='FC__dropout', lower=0.01, upper=0.4, default_value=0.02, log=True)
        FC_out_dropout = UniformFloatHyperparameter(name='FC_out_dropout', lower=0.01, upper=0.4, default_value=0.02, log=True)
        
        FC_in_Conv__NumberBlocks = UniformIntegerHyperparameter(name='FC_in_Conv__NumberBlocks', lower=1, upper=5, default_value=1,  log=False)
        FC_in_Conv__NumberLayers = UniformIntegerHyperparameter(name='FC_in_Conv__NumberLayers', lower=1, upper=5, default_value=1,  log=False)
        
        # Other hyperparameters
        lr = UniformFloatHyperparameter(name='lr', lower=1e-5, upper=1e-3, default_value=1e-4, log=True)
        LeakyReLU_alpha = UniformFloatHyperparameter(name='LeakyReLU_alpha', lower=0.01, upper=0.3, default_value=0.01, log=True)
        kernel_initializer_list = CategoricalHyperparameter(name='kernel_initializer_list', choices=['he_normal', 'glorot_uniform'])


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
    

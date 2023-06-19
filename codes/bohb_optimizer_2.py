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
  
  
# Settings the warnings to be ignored
warnings.filterwarnings('ignore')
  
# This warning won't display due to the disabled warnings
warnings.warn('Error: A warning just appeared')

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True

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
                   N_train=8192, N_valid=1024, **kwargs):

        
            super().__init__(**kwargs)

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
            
            ######### Shape of the inputs
            input1 = Input(shape=self.X1_train.shape[1:])
            input2 = Input(shape=self.X2_train.shape[1:])


            ######### Input Conv block: feed 104 feature set
            conv_inp = Conv1D(filters = config['Conv1Dinp__filters'], 
                            kernel_initializer = config['Conv1Dinp__kernel_initializer'], 
                            kernel_size = config['Conv1Dinp__kernel_size'], 
                            strides =config['Conv1Dinp__strides'], 
                            padding ='same', 
                            # kernel_regularizer=tf.keras.regularizers.l2(config['Conv1Dinp__kernel_regularizer'])  ,
                            activation='relu',
                            name='input1-Scaled')(input1)

            
            # for i in range(config['Conv1D__layers_number']):

            ######### 1st Conv Blocks
            i = 1
            Conv1D_1 = Conv1D(filters = config['Conv1D__filters']*((i+1)*2), 
                              kernel_size = config['Conv1D__kernel_size'], 
                              activation = 'relu', 
                              padding ='same', 
                              # kernel_regularizer=tf.keras.regularizers.l2(config['Conv1D__kernel_regularizer1'])  ,
                              name = 'Conv1D_layer'+str(i))(conv_inp)
            
            Max1 = MaxPooling1D(pool_size=(config['Conv1D__MaxPooling1D']),
                                name = 'MaxPooling1D_layer'+str(i))(Conv1D_1)

            
            
            ######### 2nd Conv Blocks
            i = 2
            Conv1D_2 = Conv1D(filters = config['Conv1D__filters']*((i+1)*4), 
                              kernel_size = config['Conv1D__kernel_size'], 
                              activation = 'relu', 
                              padding ='same', 
                              # kernel_regularizer=tf.keras.regularizers.l2(config['Conv1D__kernel_regularizer2'])  ,
                              name = 'Conv1D_layer'+str(i))(Max1)
            
            Max2 = MaxPooling1D(pool_size=(config['Conv1D__MaxPooling1D']),
                                name = 'MaxPooling1D_layer'+str(i))(Conv1D_2)
            
            
            ######### Input Conv block: feed MaxMin col-Scaled feature set
            input2_MinMax = Conv1D(filters = Max2.shape[-1], 
                               kernel_size = 1, 
                               padding='same',
                               activation='relu', 
                               name='input2-MinMax')(input2)

            
            ######### Concatenate the outputs of the convolutional layers
            concat = Concatenate(axis=1)([Max2, input2_MinMax])

            


            ######### Input Fully-Connected (FC) Layer
            FCinp = Dense(units = config['FCinp__units'],
                         kernel_regularizer=tf.keras.regularizers.l2(config['FCinp__kernel_regularizer']),
                         activation='relu',
                         name = 'FCinp')(concat)
            
            FCinp__Dropout = Dropout(config['FCinp__Dropout'], 
                                     name = 'FCinp__Dropout')(FCinp)
    
    
            ######### 1st FC Layer
            i = 1
            FC1 = Dense(units = config['FC__units1']*((i+1)*2),
                            kernel_regularizer=tf.keras.regularizers.l2(config['FC__kernel_regularizer1']),
                            activation='relu',
                            name = 'FC'+str(i))(FCinp__Dropout)

            FC__Dropout1 = Dropout(config['FC__Dropout'+str(i)], 
                                   name = 'FC__Dropout_layer'+str(i))(FC1)

            ######### 2nd FC Layer
            i = 2            
            FC2 = Dense(units = config['FC__units2']*((i+1)*4),
                            kernel_regularizer=tf.keras.regularizers.l2(config['FC__kernel_regularizer2']),
                            activation='relu',
                            name = 'FC'+str(i))(FC__Dropout1)

            FC__Dropout2 = Dropout(config['FC__Dropout'+str(i)], 
                                   name = 'FC__Dropout_layer'+str(i))(FC2)
            

            ######### 3rd FC Layer
            i = 3            
            FC3 = Dense(units = config['FC__units2']*((i+1)*4),
                            kernel_regularizer=tf.keras.regularizers.l2(config['FC__kernel_regularizer2']),
                            activation='relu',
                            name = 'FC'+str(i))(FC__Dropout2)

            FC__Dropout3 = Dropout(config['FC__Dropout'+str(i)], 
                                   name = 'FC__Dropout_layer'+str(i))(FC3)
            
            
            ######### FC Layers: Output
            output1 = Dense(units=(1), name='output1-gravity')(FC__Dropout3)
            output2 = Dense(units=(1), name='output2-c_o_ratio')(FC__Dropout3)
            output3 = Dense(units=(1), name='output3-metallicity')(FC__Dropout3)
            output4 = Dense(units=(1), name='output4-temperature')(FC__Dropout3)

            # Define the model with inputs and outputs
            model = Model(inputs=[input1, input2], outputs=[output1, output2, output3, output4])

            

            model.compile(loss='mae', 
                            optimizer= tf.keras.optimizers.Adam(
                            learning_rate=config['lr'],
                            )
                         )
  
            early_stop = EarlyStopping(monitor='val_loss', # Quantity to be monitored. 
                                       min_delta=0,        # Minimum change in the monitored quantity to qualify as an improvement 
                                       patience=30,        # Number of epochs with no improvement 
                                       verbose=1, 
                                       mode='auto')

            history = model.fit(x = [self.X1_train, self.X2_train], 
                                y = [self.y1_train, self.y2_train, self.y3_train, self.y4_train],  #self.x_train, self.y_train,
                              batch_size = 2**10,#config['batch_size'], # self.batch_size,
                              # validation_data=([self.X1_val, self.X2_val], [self.y1_val, self.y2_val, self.y3_val, self.y4_val]),
                              validation_split=0.1,
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
            
            print(train_score)
#             #import IPython; IPython.embed()
            return ({
                    'loss': 1-train_score[0], # remember: HpBandSter always minimizes!
                    'info': {       
                                            'test accuracy': test_score,
                                            'train accuracy': train_score,
                                            'validation accuracy': val_score,
                                            'number of parameters': model.count_params(),
            
                                    },
                    'model' : model,
                    'history' : history,

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
            # notes from Hamed:  
            # Conv1Dinp__kernel_size: # this operator 

            Conv1Dinp__filters   = CSH.UniformIntegerHyperparameter('Conv1Dinp__filters', lower=2**2, upper=2**8, default_value=2**5, log=True)
            Conv1Dinp__kernel_initializer   = CSH.CategoricalHyperparameter('Conv1Dinp__kernel_initializer', ['he_uniform','he_normal'])
            Conv1Dinp__kernel_size  = CSH.UniformIntegerHyperparameter('Conv1Dinp__kernel_size', lower= 1, upper= 64, default_value= 5, log= False) # note: 16 is a wide range
            Conv1Dinp__strides  = CSH.UniformIntegerHyperparameter('Conv1Dinp__strides', lower= 1, upper= 3, default_value= 1, log= False)
            Conv1Dinp__AveragePooling1D      = CSH.UniformIntegerHyperparameter('Conv1Dinp__AveragePooling1D', lower= 1, upper= 7, default_value= 1, log=False) # note: 64 is a wide range
            Conv1Dinp__dropout = CSH.UniformFloatHyperparameter('Conv1Dinp__dropout', lower=0.01, upper=0.7, default_value=0.1, log=True)
            Conv1Dinp__kernel_regularizer = CSH.UniformFloatHyperparameter('Conv1Dinp__kernel_regularizer', lower=0.001, upper=0.02, default_value=0.003/2., log=True)
            cs.add_hyperparameters([Conv1Dinp__filters, 
                                    Conv1Dinp__kernel_initializer, 
                                    Conv1Dinp__kernel_size,
                                    Conv1Dinp__strides,
                                    Conv1Dinp__AveragePooling1D,
                                    Conv1Dinp__dropout,
                                    Conv1Dinp__kernel_regularizer,
                                   ])
                         
                          
                    
            Conv1D__layers_number  = CSH.UniformIntegerHyperparameter('Conv1D__layers_number', lower= 2, upper= 5, default_value= 2, log= False)
            Conv1D__filters   = CSH.UniformIntegerHyperparameter('Conv1D__filters', lower=2**2, upper=2**8, default_value=2**5, log=True)
            Conv1D__kernel_size  = CSH.UniformIntegerHyperparameter('Conv1D__kernel_size', lower= 1, upper= 64, default_value= 5, log= False)
            Conv1D__strides  = CSH.UniformIntegerHyperparameter('Conv1D__strides', lower= 1, upper= 6, default_value= 1, log= False)
            Conv1D__kernel_regularizer1 = CSH.UniformFloatHyperparameter('Conv1D__kernel_regularizer1', lower=0.001, upper=0.02, default_value=0.003/2., log=True)
            Conv1D__kernel_regularizer2 = CSH.UniformFloatHyperparameter('Conv1D__kernel_regularizer2', lower=0.001, upper=0.02, default_value=0.003/2., log=True)
            Conv1D__dropout_1 = CSH.UniformFloatHyperparameter('Conv1D__dropout_1', lower=0.01, upper=0.7, default_value=0.1, log=True)
            Conv1D__dropout_2 = CSH.UniformFloatHyperparameter('Conv1D__dropout_2', lower=0.01, upper=0.7, default_value=0.1, log=True)
            Conv1D__MaxPooling1D      = CSH.UniformIntegerHyperparameter('Conv1D__MaxPooling1D', lower= 1, upper= 6, default_value= 1, log=False)
            
            cs.add_hyperparameters([Conv1D__layers_number, 
                                    Conv1D__filters, 
                                    Conv1D__kernel_size, 
                                    Conv1D__strides,
                                    Conv1D__kernel_regularizer1,
                                    Conv1D__kernel_regularizer2,
                                    Conv1D__dropout_1,
                                    Conv1D__dropout_2,
                                    Conv1D__MaxPooling1D
                                    ])

            
            
            FCinp__units   = CSH.UniformIntegerHyperparameter('FCinp__units', lower=2**3, upper=2**8, default_value=2**5, log=True)
            FCinp__kernel_regularizer = CSH.UniformFloatHyperparameter('FCinp__kernel_regularizer', lower=0.001, upper=0.02, default_value=0.003/2., log=True)
            FCinp__Dropout = CSH.UniformFloatHyperparameter('FCinp__Dropout', lower=0.01, upper=0.7, default_value=0.1, log=True)
            cs.add_hyperparameters([FCinp__units, 
                                    FCinp__kernel_regularizer, 
                                    FCinp__Dropout,
                                    ])           
                          

            
            FC__layers_number = CSH.UniformIntegerHyperparameter('FC__layers_number', lower=4, upper=5, default_value=4, log=False)
            FC__units1   = CSH.UniformIntegerHyperparameter('FC__units1', lower=2**3, upper=2**8, default_value=2**5, log=True)
            FC__units2   = CSH.UniformIntegerHyperparameter('FC__units2', lower=2**3, upper=2**8, default_value=2**5, log=True)
            FC__kernel_regularizer1 = CSH.UniformFloatHyperparameter('FC__kernel_regularizer1', lower=0.001, upper=0.02, default_value=0.003/2., log=True)
            FC__kernel_regularizer2 = CSH.UniformFloatHyperparameter('FC__kernel_regularizer2', lower=0.001, upper=0.02, default_value=0.003/2., log=True)
            FC__kernel_regularizer3 = CSH.UniformFloatHyperparameter('FC__kernel_regularizer3', lower=0.001, upper=0.02, default_value=0.003/2., log=True)
            FC__optimizer   = CSH.CategoricalHyperparameter('FC__optimizer', ['adam','RMSprop'])
            FC__Dropout1 = CSH.UniformFloatHyperparameter('FC__Dropout1', lower=0.01, upper=0.7, default_value=0.1, log=True)
            FC__Dropout2 = CSH.UniformFloatHyperparameter('FC__Dropout2', lower=0.01, upper=0.7, default_value=0.1, log=True)
            FC__Dropout3 = CSH.UniformFloatHyperparameter('FC__Dropout3', lower=0.01, upper=0.7, default_value=0.1, log=True)
            # batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=2**6, upper=2**10, default_value=2**8, log=False)
            batch_size = CSH.CategoricalHyperparameter('batch_size', [2**6, 2**7, 2**8, 2**9, 2**10])

            
            lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-3', log=True)

            cs.add_hyperparameters([FC__layers_number, 
                                    FC__units1, 
                                    FC__units2, 
                                    FC__kernel_regularizer1,
                                    FC__kernel_regularizer2,
                                    FC__kernel_regularizer3,
                                    FC__optimizer, 
                                    batch_size,
                                    lr,
                                    FC__Dropout1,
                                    FC__Dropout2,
                                    FC__Dropout3,
                                    
                                   ])


            return cs
        
        
 
                
  
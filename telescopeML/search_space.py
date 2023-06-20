# search_space.py
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter
import  ConfigSpace as CSH

# def get_search_space():
#     cs = ConfigurationSpace()

#     lr = UniformFloatHyperparameter("learning_rate", 0.001, 0.1)
#     optimizer = CategoricalHyperparameter("optimizer", ["adam", "sgd"])
#     num_layers = UniformIntegerHyperparameter("num_layers", 1, 5)
#     dropout = UniformFloatHyperparameter("dropout", 0.0, 0.5)

#     cs.add_hyperparameters([lr, optimizer, num_layers, dropout])

#     return cs

def get_configspace2():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = ConfigurationSpace()
        # notes from Hamed:  
        # Conv1Dinp__kernel_size: # this operator 
        # Kenrel size is the same, Stride not 
        # Conv1Dinp__filters: 1 filter, 2*, 4*, .... Optimize this once
        # Stride = 1
        # Optimize size
        # kernel_regularizer = Default
        # MaxPooling1D: 

        Conv__layers_number  = CSH.UniformIntegerHyperparameter('Conv__layers_number', lower= 2, upper= 5, default_value= 2, log= False)
        Conv__GaussianNoise = CSH.CategoricalHyperparameter('Conv__GaussianNoise', [0.01, 0.02, 0.03, 0.04, 0.05, ])
        Conv__filters   = CSH.UniformIntegerHyperparameter('Conv__filters', lower=4, upper=16, default_value=6, log=False) # 
        Conv__kernel_initializer   = CSH.CategoricalHyperparameter('Conv__kernel_initializer', ['he_uniform','he_normal'])
        Conv__kernel_size  = CSH.UniformIntegerHyperparameter('Conv__kernel_size', lower= 1, upper= 16, default_value= 5, log= False) # note: 16 is a wide range
        Conv__MaxPooling1D      = CSH.UniformIntegerHyperparameter('Conv__MaxPooling1D', lower= 1, upper= 6, default_value= 1, log=False)
        FC__units   = CSH.UniformIntegerHyperparameter('FC__units', lower=64, upper=512, default_value=100, log=False)
        FC_in_Conv__units   = CSH.UniformIntegerHyperparameter('FC_in_Conv__units', lower=16, upper=512, default_value=100, log=False)

        cs.add_hyperparameters([
                                Conv__GaussianNoise, 
                                Conv__filters,
                                Conv__kernel_initializer, 
                                Conv__kernel_size,
                                FC__units,
                                FC_in_Conv__units,
                                Conv__MaxPooling1D,
                                Conv__layers_number,
                               ])


        return cs
    
    

def get_configspace_NOT_USED():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = ConfigurationSpace()
        # notes from Hamed:  
        # Conv1Dinp__kernel_size: # this operator 
        # Kenrel size is the same, Stride not 
        # Conv1Dinp__filters: 1 filter, 2*, 4*, .... Optimize this once
        # Stride = 1
        # Optimize size
        # kernel_regularizer = Default
        # MaxPooling1D: 

        Conv1D__GaussianNoise = CSH.CategoricalHyperparameter('Conv1D__GaussianNoise', [0.01, 0.02, 0.03, 0.04, 0.05, ])
        
        
        Conv1Dinp__filters   = CSH.UniformIntegerHyperparameter('Conv1Dinp__filters', lower=4, upper=16, default_value=14, log=False) # 
        Conv1Dinp__kernel_initializer   = CSH.CategoricalHyperparameter('Conv1Dinp__kernel_initializer', ['he_uniform','he_normal'])
        Conv1Dinp__kernel_size  = CSH.UniformIntegerHyperparameter('Conv1Dinp__kernel_size', lower= 1, upper= 16, default_value= 5, log= False) # note: 16 is a wide range
        Conv1Dinp__strides  = CSH.UniformIntegerHyperparameter('Conv1Dinp__strides', lower= 1, upper= 3, default_value= 1, log= False)
        Conv1Dinp__AveragePooling1D      = CSH.UniformIntegerHyperparameter('Conv1Dinp__AveragePooling1D', lower= 1, upper= 7, default_value= 1, log=False) # note: 64 is a wide range
        Conv1Dinp__dropout = CSH.UniformFloatHyperparameter('Conv1Dinp__dropout', lower=0.01, upper=0.7, default_value=0.1, log=True)
        Conv1Dinp__kernel_regularizer = CSH.UniformFloatHyperparameter('Conv1Dinp__kernel_regularizer', lower=0.001, upper=0.02, default_value=0.003/2., log=True)
        cs.add_hyperparameters([
                                Conv1Dinp__filters, 
                                Conv1Dinp__kernel_initializer, 
                                Conv1Dinp__kernel_size,
                                Conv1Dinp__strides,
                                Conv1Dinp__AveragePooling1D,
                                Conv1Dinp__dropout,
                                Conv1Dinp__kernel_regularizer,
                                Conv1D__GaussianNoise,
                               ])



        Conv1D__layers_number  = CSH.UniformIntegerHyperparameter('Conv1D__layers_number', lower= 2, upper= 5, default_value= 2, log= False)
        Conv1D__filters1   = CSH.UniformIntegerHyperparameter('Conv1D__filters1', lower=14, upper=38, default_value=30, log=False)
        Conv1D__filters2   = CSH.UniformIntegerHyperparameter('Conv1D__filters2', lower=2**2, upper=2**8, default_value=2**5, log=True)
        Conv1D__kernel_size  = CSH.UniformIntegerHyperparameter('Conv1D__kernel_size', lower= 1, upper= 64, default_value= 5, log= False)
        Conv1D__strides  = CSH.UniformIntegerHyperparameter('Conv1D__strides', lower= 1, upper= 6, default_value= 1, log= False)
        Conv1D__kernel_regularizer1 = CSH.UniformFloatHyperparameter('Conv1D__kernel_regularizer1', lower=0.001, upper=0.02, default_value=0.003/2., log=True)
        Conv1D__kernel_regularizer2 = CSH.UniformFloatHyperparameter('Conv1D__kernel_regularizer2', lower=0.001, upper=0.02, default_value=0.003/2., log=True)
        Conv1D__dropout_1 = CSH.UniformFloatHyperparameter('Conv1D__dropout_1', lower=0.01, upper=0.7, default_value=0.1, log=True)
        Conv1D__dropout_2 = CSH.UniformFloatHyperparameter('Conv1D__dropout_2', lower=0.01, upper=0.7, default_value=0.1, log=True)
        Conv1D__MaxPooling1D_1      = CSH.UniformIntegerHyperparameter('Conv1D__MaxPooling1D_1', lower= 1, upper= 6, default_value= 1, log=False)
        Conv1D__MaxPooling1D_2      = CSH.UniformIntegerHyperparameter('Conv1D__MaxPooling1D_2', lower= 1, upper= 6, default_value= 1, log=False)
        FC__units_Conv1D   = CSH.UniformIntegerHyperparameter('FC__units_Conv1D', lower=2**3, upper=2**8, default_value=2**5, log=True)

        cs.add_hyperparameters([
                                Conv1D__layers_number, 
                                Conv1D__filters1, 
                                Conv1D__filters2,
                                Conv1D__kernel_size, 
                                Conv1D__strides,
                                Conv1D__kernel_regularizer1,
                                Conv1D__kernel_regularizer2,
                                Conv1D__dropout_1,
                                Conv1D__dropout_2,
                                Conv1D__MaxPooling1D_1,
                                Conv1D__MaxPooling1D_2,
                                FC__units_Conv1D
                                ])



        FCinp__units   = CSH.UniformIntegerHyperparameter('FCinp__units', lower=2**3, upper=2**8, default_value=2**5, log=True)
        FCinp__kernel_regularizer = CSH.UniformFloatHyperparameter('FCinp__kernel_regularizer', lower=0.001, upper=0.02, default_value=0.003/2., log=True)
        FCinp__Dropout = CSH.UniformFloatHyperparameter('FCinp__Dropout', lower=0.01, upper=0.7, default_value=0.1, log=True)
        cs.add_hyperparameters([
                                # FCinp__units, 
                                # FCinp__kernel_regularizer, 
                                # FCinp__Dropout,
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
        batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=2**6, upper=2**10, default_value=2**8, log=False)
        batch_size = CSH.CategoricalHyperparameter('batch_size', [2**6, 2**7, 2**8, 2**9, 2**10])


        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-4', log=True)

        cs.add_hyperparameters([
                                # FC__layers_number, 
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
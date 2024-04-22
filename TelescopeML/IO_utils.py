import os
import pickle as pk
from tensorflow.keras.models import load_model

# from tensorflow.keras.models import model_from_json

__reference_data__ = os.getenv("TelescopeML_reference_data")
# print(__reference_data__)



class LoadSave:
    """
    Load and Save trained operators, models, and datasets
    """

    def __init__(self,
                 trained_ML_model_name,
                 ml_method,
                 is_feature_improved = None,
                 is_augmented = None,
                 is_tuned = None,
                 ):
        self.trained_ML_model_name = trained_ML_model_name
        self.ml_method = ml_method
        self.is_feature_improved = is_feature_improved
        self.is_augmented = is_augmented
        self.is_tuned = is_tuned
        self.base_path = os.path.join(__reference_data__,'trained_ML_models')
        # '../outputs/trained_models/'

    def create_generic_path(self, output_indicator):
        """
        Create the generic path for saving or loading the trained model

        Inputs:
        -------
        - indicator (str): Indicator for the model type

        Returns:
        --------
        - generic_path (str): The generic path for saving or loading the model
        """
        file_name = f'{output_indicator}__{self.trained_ML_model_name}__' \
                    f'{self.ml_method}'

        generic_path = os.path.join(self.base_path, file_name)
        return generic_path

    def load_or_dump_trained_object(self,
                                    trained_object,
                                    output_indicator,
                                    load_or_dump='dump'):
        """
        Load or save the trained object

        Inputs:
        -------
        - trained_object : The object to be saved or loaded
        - indicator (str): Indicator for the type of trained object
        - load_or_dump (str): 'dump' or 'load'
        """
        generic_path = self.create_generic_path(output_indicator)

        if load_or_dump == 'dump':
            with open(generic_path, 'wb') as file:
                pk.dump(trained_object, file)
        elif load_or_dump == 'load':
            with open(generic_path, 'rb') as file:
                return pk.load(file)


def load_or_dump_trained_model_CNN(
                                   trained_model = None,
                                   output_indicator='TrainedCNN',
                                   load_or_dump='dump'):

    path_architecture = os.path.join(__reference_data__,
                         'trained_ML_models/trained_CNN_architecture_'+output_indicator+'.h5',
                         )

    path_history = os.path.join(__reference_data__,
                         'trained_ML_models/trained_CNN_history_'+output_indicator+'.pkl',
                         )

    path_weights = os.path.join(__reference_data__,
                         'trained_ML_models/trained_CNN_weights_'+output_indicator+'.h5',
                         )

    """
    Load or save the trained CNN model

    Inputs:
    -------
    - trained_model : The trained CNN model
    - indicator (str): Indicator for the type of trained model
    - load_or_dump (str): 'dump' or 'load'
    """
    # json_path = self.create_generic_path(f'{indicator}_json')
    # weights_path = self.create_generic_path(f'{indicator}_weights')

    if load_or_dump == 'dump':
        trained_model.trained_ML_model.save(path_architecture)
        trained_model.trained_ML_model.save_weights(path_weights)

        with open(path_history, 'wb') as file:
            pk.dump(trained_model.history.history, file)

    elif load_or_dump == 'load':
        loaded_model = load_model(path_architecture)
        loaded_model.load_weights(path_weights)

        # Loading the saved history object
        with open(path_history, 'rb') as file:
                history = pk.load(file)

        return loaded_model, history


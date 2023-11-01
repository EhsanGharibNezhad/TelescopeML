import os
import pickle as pk
from tensorflow.keras.models import load_model

# from tensorflow.keras.models import model_from_json

__reference_data__ = os.getenv("TelescopeML_reference_data")
print(__reference_data__)
#
# if __reference_data__ is None:
#     raise Exception('\n'
#                        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
#                        "TelescopeML Error Message: \n\n"
#                        "You need to define the path to your reference data.\n"
#                        "Check out this tutorial: https://ehsangharibnezhad.github.io/TelescopeML/installation.html\n"
#                        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#                     )
# else:
#     pass


class LoadSave:
    """
    Load and Save trained operators, models, and datasets
    """

    def __init__(self,
                 ml_model_str,
                 ml_method,
                 is_feature_improved,
                 is_augmented,
                 is_tuned,
                 ):
        self.ml_model_str = ml_model_str
        self.ml_method = ml_method
        self.is_feature_improved = is_feature_improved
        self.is_augmented = is_augmented
        self.is_tuned = is_tuned
        self.base_path = os.path.join(__reference_data__,'trained_ML_models')
        # '../outputs/trained_models/'

    def create_generic_path(self, indicator):
        """
        Create the generic path for saving or loading the trained model

        Inputs:
        -------
        - indicator (str): Indicator for the model type

        Returns:
        --------
        - generic_path (str): The generic path for saving or loading the model
        """
        file_name = f'{indicator}__{self.ml_model_str}' \
                    f'__Is_feature_improved_{self.is_feature_improved}' \
                    f'__Is_augmented_{self.is_augmented}__' \
                    f'Is_tuned__{self.is_tuned}__' \
                    f'{self.ml_method}'
        generic_path = os.path.join(self.base_path, file_name)
        return generic_path

    def load_or_dump_trained_object(self,
                                    trained_object,
                                    indicator,
                                    load_or_dump='dump'):
        """
        Load or save the trained object

        Inputs:
        -------
        - trained_object : The object to be saved or loaded
        - indicator (str): Indicator for the type of trained object
        - load_or_dump (str): 'dump' or 'load'
        """
        generic_path = self.create_generic_path(indicator)

        if load_or_dump == 'dump':
            with open(generic_path, 'wb') as file:
                pk.dump(trained_object, file)
        elif load_or_dump == 'load':
            with open(generic_path, 'rb') as file:
                return pk.load(file)


def load_or_dump_trained_model_CNN(
                                   trained_model = None,
                                   indicator='TrainedCNN',
                                   load_or_dump='dump'):

    path_architecture = os.path.join(__reference_data__,
                         'trained_ML_models/trained_CNN_architecture_'+indicator+'.h5',
                         )

    path_history = os.path.join(__reference_data__,
                         'trained_ML_models/trained_CNN_history_'+indicator+'.pkl',
                         )

    path_weights = os.path.join(__reference_data__,
                         'trained_ML_models/trained_CNN_weights_'+indicator+'.h5',
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
        trained_model.trained_model.save(path_architecture)
        trained_model.trained_model.save_weights(path_weights)

        with open(path_history, 'wb') as file:
            pk.dump(trained_model.history.history, file)

    elif load_or_dump == 'load':
        loaded_model = load_model(path_architecture)
        loaded_model.load_weights(path_weights)

        # Loading the saved history object
        with open(path_history, 'rb') as file:
                history = pk.load(file)

        return loaded_model, history

# class LoadSave:
#     """
#     Load and Save trained operators, models, and datasets
#     """
#
#     def __init__(self,
#                  ml_model_str,
#                  ml_method,
#                  is_feature_improved,
#                  is_augmented,
#                  is_tuned,
#                  ):
#         self.ml_model_str = ml_model_str
#         self.ml_method = ml_method
#         self.is_feature_improved = is_feature_improved
#         self.is_augmented = is_augmented
#         self.is_tuned = is_tuned
#         self.reference_data = __reference_data__
#
#     def create_generic_path(self, indicator):
#         """
#         Create the generic path for saving or loading the trained model
#
#         Inputs:
#         -------
#         - indicator (str): Indicator for the model type
#
#         Returns:
#         --------
#         - generic_path (str): The generic path for saving or loading the model
#         """
#         file_name = f'{indicator}__{self.ml_model_str}' \
#                     f'__Is_feature_improved_{self.is_feature_improved}' \
#                     f'__Is_augmented_{self.is_augmented}__' \
#                     f'Is_tuned__{self.is_tuned}__' \
#                     f'{self.ml_method}'
#         generic_path = os.path.join(self.reference_data, file_name)
#         return generic_path
#
#     def load_or_dump_trained_object(self,
#                                     trained_object,
#                                     indicator,
#                                     load_or_dump='dump'):
#         """
#         Load or save the trained object
#
#         Inputs:
#         -------
#         - trained_object : The object to be saved or loaded
#         - indicator (str): Indicator for the type of trained object
#         - load_or_dump (str): 'dump' or 'load'
#         """
#         generic_path = self.create_generic_path(indicator)
#
#         if load_or_dump == 'dump':
#             with open(generic_path, 'wb') as file:
#                 pk.dump(trained_object, file)
#         elif load_or_dump == 'load':
#             with open(generic_path, 'rb') as file:
#                 return pk.load(file)
#
#
# def load_or_dump_trained_model_CNN(
#                                    trained_model = None,
#                                    indicator='TrainedCNN',
#                                    load_or_dump='dump'):
#     """
#     Load or save the trained CNN model
#
#     Inputs:
#     -------
#     - trained_model : The trained CNN model
#     - indicator (str): Indicator for the type of trained model
#     - load_or_dump (str): 'dump' or 'load'
#     """
#     # json_path = self.create_generic_path(f'{indicator}_json')
#     # weights_path = self.create_generic_path(f'{indicator}_weights')
#
#     if load_or_dump == 'dump':
#         trained_model.trained_model.save(
#             os.path.join(__reference_data__,
#                          'trained_ML_models/trained_CNN_architecture_'+indicator+'.h5'
#                          ))
#
#         trained_model.trained_model.save_weights(
#             os.path.join(__reference_data__,
#                          'trained_ML_models/trained_CNN_weights_'+indicator+'.h5',
#                          ))
#
#
#         with open(
#                 os.path.join(__reference_data__,
#                              'trained_ML_models/trained_CNN_history_'+indicator+'.h5',
#                              ),
#             'wb') as file:
#
#         # with open(
#         #         f'../outputs/trained_models/trained_CNN_history_{indicator}.pkl',
#         #         'wb') as file:
#             pk.dump(trained_model.history.history, file)
#
#     elif load_or_dump == 'load':
#
#         loaded_model = load_model(
#             os.path.join(__reference_data__,
#                          'trained_ML_models/trained_CNN_architecture_'+indicator+'.h5' ))
#
#         loaded_model.load_weights(
#             os.path.join(__reference_data__,
#                          'trained_ML_models/trained_CNN_weights_'+indicator+'.h5' ))
#
#         # loaded_model = load_model(
#         #     f'../outputs/trained_models/trained_CNN_architecture_{indicator}.h5')
#         # loaded_model.load_weights(
#         #     f'../outputs/trained_models/trained_CNN_weights_{indicator}.h5')
#
#         # Loading the saved history object
#
#         with open(
#                 os.path.join(__reference_data__,
#                              'trained_ML_models/trained_CNN_weights_'+indicator+'.pkl'
#                              ),
#                 'rb') as file:
#                 history = pk.load(file)
#
#         # with open(
#         #         f'../outputs/trained_models/trained_CNN_history_{indicator}.pkl',
#         #         'rb') as file:
#         #         history = pk.load(file)
#
#         return loaded_model, history

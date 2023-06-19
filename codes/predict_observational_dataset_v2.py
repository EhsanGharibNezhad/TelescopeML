# Import functions from other modules ============================

from io_funs import LoadSave
# from train_ml_regression_2 import TrainMlRegression

# Import python libraries ========================================

# Dataset manipulation libraries
import pandas as pd
import numpy as np
import pickle as pk
import os

# ML algorithm libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Data Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

from bokeh.io import output_notebook
from bokeh.layouts import row, column
output_notebook()
from bokeh.plotting import show,figure
TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
]
from skopt.plots import plot_evaluations

# check later if you need this
# from bokeh.palettes import Category20, colorblind
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import warnings
# warnings.filterwarnings('ignore')

# from bokeh.palettes import Category20, colorblind
from tensorflow.keras.models import Sequential, model_from_json
#
# color_list = sns.color_palette("colorblind", 30)
# palette_list = sns.color_palette("colorblind", 30)





from scipy.interpolate import interp1d
import os
import scipy.stats as st
from sklearn.preprocessing import StandardScaler


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
# Data Visualizing libararies
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
from bokeh.io import output_notebook
from bokeh.layouts import row, column
output_notebook()
from bokeh.plotting import show,figure
TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
]

import astropy.units as u
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import pchip
from tensorflow.keras.models import Sequential, model_from_json
import spectres

# ===============================================================================
# ========================                           ============================
# ========================    observational_spectra  ============================
# ========================                           ============================
# ===============================================================================


class ProcessObservationalDataset:
    """
    Load, process, visualize observational spectra

    Attribute
    ---------
    - feature_values_obs: Fluxes for each feature (wavelength) from observational data
    - feature_names_obs: Name of features (wavelength) from observational data, e.g., 0.9, 1.0, 1.1 micron
    - feature_names_synthetic: Name of features (wavelength) from synthetic data
    """
    def __init__(self, feature_values_obs, feature_names_obs, feature_names_synthetic):
        self.feature_values_obs = feature_values_obs
        self.feature_names_obs = feature_names_obs
        self.feature_names_model = np.sort(feature_names_synthetic)
        self.Fnu_obs = self.Flam2Fnu()  # Convert Flam_obs to Fnu_obs

    def Flam2Fnu(self):
        """
        Convert F_lambda to F_nu
        """
        fnu = (self.feature_values_obs * u.erg / u.s / u.Angstrom / u.cm ** 2).to(
            u.erg / u.s / u.cm ** 2 / u.Hz, equivalencies=u.spectral_density(self.feature_names_obs * u.micron))
        return fnu.to_value()

    
    def bd_info(self, bd_name, bd_Teff, bd_logg, bd_met, bd_distance_pc, bd_radius_Rjup):
        """
        Set up and save the brown dwarf (bd) literature data

        Parameters
        -----------
        - bd_name: Brown dwarf name
        - bd_Teff: Brown dwarf effective temperature
        - bd_logg: Brown dwarf log(surface gravity) in cgs
        - bd_met: Brown dwarf metallicity
        - bd_distance_pc: Distance of the object from Earth in pc unit
        - bd_radius_Rjup: Radius of the object in Jupiter radius

        Returns
        --------
        - Calculate Fnu_obs_absolute from Fnu_obs
        """
        self.bd = {}
        self.bd['bd_name'] = bd_name
        self.bd['bd_Teff'] = bd_Teff
        self.bd['bd_logg'] = bd_logg
        self.bd['bd_met'] = bd_met
        self.bd['bd_distance_pc'] = bd_distance_pc
        self.bd['bd_radius_Rjup'] = bd_radius_Rjup
        self.Fnu_obs_absolute = self.Fnu_obs * (
                self.bd['bd_distance_pc'] * ((u.pc).to(u.jupiterRad)) / (self.bd['bd_radius_Rjup'])) ** 2

    def flux_interpolated(self, print_results=False, plot_results=True, use_spectres=False):
        if use_spectres:
            self.Fnu_obs_absolute_intd = spectres.spectres(self.feature_names_model,
                                                           np.float128(self.feature_names_obs),
                                                           np.float128(self.Fnu_obs_absolute))
        else:
            flux_intd = pchip(self.feature_names_obs, self.Fnu_obs_absolute)
            self.Fnu_obs_absolute_intd = flux_intd(self.feature_names_model)

        self.df_flux_object = pd.DataFrame(list(self.Fnu_obs_absolute_intd),
                                           index=[str(x) for x in np.round(self.feature_names_model, 3)]).T

        if print_results:
            print('---    Object Flux     ----')
            display(self.df_flux_object)
            print('---    Object Flux Interpolate     ----')
            display(pd.DataFrame(self.Fnu_obs_absolute_intd))

        if plot_results:
            self.plot_observed()

    def plot_observed(self):
        """
        Plot the observed telescope dataset
        """
        p = figure(width=1000, height=300, y_axis_type="log", x_axis_type="linear")

        p.line(self.feature_names_obs,
               self.Fnu_obs_absolute,
               color='green',
               legend_label='Fnu_obs_absolute')

        p.scatter(np.float32(self.df_flux_object.columns.values),
                  self.df_flux_object.values[0],
                  color='blue',
                  legend_label='Fnu_obs_absolute_intd')

        p.xaxis.axis_label = 'Wavelength  [𝜇m]'
        p.yaxis.axis_label = 'Flux (Fν)'

        # Set the font size for axis labels and tick labels
        p.xaxis.axis_label_text_font_size = '12pt'
        p.yaxis.axis_label_text_font_size = '12pt'
        p.xaxis.major_label_text_font_size = '11pt'
        p.yaxis.major_label_text_font_size = '11pt'

        show(p)


class ProcessObservationalDataset_oldVesion1:
    """
    Load, process, visualize observational spectra

    Attribite
    ---------
    - feature_values_obs : Flam_obs = fluxes for each feature (wavelength) from observational data
    - feature_names_obs : name of features (wavelength) from observational data,; e.g., 0.9, 1.0, 1.1 micron
    - feature_names_synthetic : name of features (wavelength) from synthetic data
    """
    def __init__(self, 
                 feature_values_obs,
                 feature_names_obs,
                 feature_names_synthetic,
                ):
        self.feature_values_obs = feature_values_obs 
        self.feature_names_obs = feature_names_obs     
        self.feature_names_model = np.sort(feature_names_synthetic)
        self.Fnu_obs = self.Flam2Fnu() # Fnu_obs: convert Flam_obs to Fnu_obs
        
        
    def Flam2Fnu(self):
        """
        Convert F_lambda to F_nu
        """
        fnu = ( self.feature_values_obs * u.erg / u.s / u.Angstrom / u.cm**2 ).to(u.erg / u.s / u.cm**2 / u.Hz, equivalencies = u.spectral_density( self.feature_names_obs * u.micron ) )
        return fnu.to_value() 


    def bd_info(self, bd_name, bd_Teff, bd_logg, bd_met, bd_distance_pc, bd_radius_Rjup):
        """
        dic to setup and save the brown dwarf (e.g., bd) literature data

        Parameters
        -----------
            - bd_name: brown dwarf name
            - bd_Teff: brown dwarf effective temperature
            - bd_logg: brown dwarf log(surface gravity) in cgs
            - bd_met: brown dwarf metallicity
            - bd_distance_pc: the distance of the object from Earth in pc unit
            - bd_radius_Rjup: the radius of the object in Jupiter radius
            # Note: 1 Jupiter radius = 2.26566120904345E-09 Parsec

        Returns
        --------
            - Calculate Fnu_obs_absolute from Fnu_obs
        """
        self.bd = {}
        self.bd['bd_name']=bd_name
        self.bd['bd_Teff']=bd_Teff
        self.bd['bd_logg']=bd_logg
        self.bd['bd_met'] = bd_met
        self.bd['bd_distance_pc']=bd_distance_pc
        self.bd['bd_radius_Rjup'] = bd_radius_Rjup
        self.Fnu_obs_absolute = self.Fnu_obs * ( self.bd['bd_distance_pc']*((u.pc).to(u.jupiterRad)) / (self.bd['bd_radius_Rjup']) )**2 # Flam_abs = (Distance / Radius)

        
    def flux_interpolated( self,
                           print_results = False,
                           plot_results = True,
                           use_spectres = False,
                          ):
        
        if use_spectres == True:
            self.Fnu_obs_absolute_intd = spectres.spectres(self.feature_names_model, 
                              np.float128(self.feature_names_obs),
                              np.float128(self.Fnu_obs_absolute))
        elif use_spectres == False:
            
            flux_intd = pchip(self.feature_names_obs , self.Fnu_obs_absolute)
            self.Fnu_obs_absolute_intd = flux_intd(self.feature_names_model) # interpolated F_nu           

        
        self.df_flux_object = pd.DataFrame(list(self.Fnu_obs_absolute_intd), 
                                           index=[str(x) for x in np.round(self.feature_names_model, 3)]).T
        
        if print_results:
            print('---    Object Flux     ----')
            display( self.df_flux_object )
            print('---    Object Flux Interpolate     ----')
            display( pd.DataFrame(self.Fnu_obs_absolute_intd) )
        
        if plot_results:
            self.plot_observed()
        
        
        
    def plot_observed(self):
        """
        plot the observed telescope dataset
        """
        p = figure( width=1000, height=300,
                    y_axis_type="log", x_axis_type="linear",)

        p.line( self.feature_names_obs,
                self.Fnu_obs_absolute,
                color='green',
                legend_label = 'Fnu_obs_absolute')
        
        p.scatter( np.float32( self.df_flux_object.columns.values ),
                   self.df_flux_object.values[0],
                   color='blue',
                   legend_label = 'Fnu_obs_absolute_intd')

        p.xaxis.axis_label = 'Wavelength  [𝜇m]'
        p.yaxis.axis_label = 'Flux (Fν)'

        # Set the font size for axis labels and tick labels
        p.xaxis.axis_label_text_font_size = '14pt'
        p.yaxis.axis_label_text_font_size = '14pt'
        p.xaxis.major_label_text_font_size = '12pt'
        p.yaxis.major_label_text_font_size = '12pt'
    
        show(p)

 
        
        
# ===============================================================================
# ===============                                               =================
# ===============     predict_obs_parameters_from_ML_model      =================
# ===============                                               =================
# ===============================================================================
       
    
class PredictObsParametersRegression:

    """
    Predict targets of observational data utilizing the trained regression models

    Inputs
    -------
        - trained_model
        - feature_values (float): set of values associated with the features/attributes in the dataset, e.g., flux values
        - feature_names (str): name of features/attributes; e.g., wavelengths
        - target_values (float): set of values associated with the target features/attributes in the dataset
        - target_name (str): name of target features/attributes; e.g., temperature, gravity
        - is_tuned: Hyperparameter tuning: 'yes' or 'no'
        - wl (float): wavelengths
        
      Control Options:
        - is_tuned (str): Are hyper-parameters tuned via GridSearchCV? yes or no
        - param_grid (dic): a list of hyper-parameters to be optimized as well as their range of values
                            Note: param_grid is used is is_tuned = 'yes'
        - spectral_resolution: resolution of the training/example dataset
        - is_feature_improved (str): if yes-> type the method; e.g, RFE, PCA, ..., IF not -> type 'no'          

    """
    
    def __init__(self, 
                 trained_model,
                 feature_names,          
                 target_name,    
                 
                 is_tuned,               
                 is_feature_improved,   
                 is_augmented,
                 ml_model_str,
                 
                 obs_spectra,
                 # trained_StandardScaler_X,
                 # trained_StandardScaler_y,
                 # trained_pca_model,
                 # features2drop_str,

                ):
        

            
        
        self.feature_names = feature_names 
        self.target_name   = target_name 
        self.is_tuned      = is_tuned 
        self.is_feature_improved = is_feature_improved
        
        self.is_augmented  =  is_augmented
        self.ml_model_str  =  ml_model_str
        
        self.obs_spectra = obs_spectra#.reshape(1, -1)

        if trained_model == None:
            trained_model = LoadSave(self.ml_model_str,
                     self.is_feature_improved,
                     self.is_augmented,
                     self.is_tuned,).load_or_dump_trained_object(trained_object = None,
                                                                indicator = 'TrainedModel',
                                                                load_or_dump = 'load')
        self.trained_model = trained_model


    def StandardScaler_X(self,
                         indicator='StandardScaler_X'):
        """
        Utilize the trained StandardScaler and apply them on  X_train and X_test
        """
        StandardScaler_X = LoadSave(self.ml_model_str,
                                    self.is_feature_improved,
                                    self.is_augmented,
                                    self.is_tuned, ).load_or_dump_trained_object(trained_object=None,
                                                                                 indicator=indicator,
                                                                                 load_or_dump='load')

        self.StandardScaler_X = StandardScaler_X

        if indicator == 'StandardScaler_row_X':
            SS = StandardScaler()
            self.X_train = SS.fit_transform(self.T).T

        elif indicator == 'StandardScaler_col_X':
            self.X_train = StandardScaler_X.transform(self.X_train)
            self.X_test = StandardScaler_X.transform(self.X_test)

    def StandardScaler_y(self,
                         indicator='StandardScaler_y'):
        """
        Utilize the trained StandardScaler and apply them on  y_train and y_test
        """
        StandardScaler_y = LoadSave(self.ml_model_str,
                                    self.is_feature_improved,
                                    self.is_augmented,
                                    self.is_tuned, ).load_or_dump_trained_object(trained_object=None,
                                                                                 indicator=indicator,
                                                                                 load_or_dump='load')

        self.StandardScaler_y = StandardScaler_y

    # #  =======  transform_StandardScaler_obs_spectra =================================================
    #
    def transform_StandardScaler_obs_spectra(self, load_operator = False):
        if load_operator == False:
            ss = StandardScaler()
            self.obs_spectra_scaled =  ss.fit_transform(self.obs_spectra.values.T).T
        elif load_operator == True:
            self.obs_spectra_scaled =  self.trained_StandardScaler_X.transform( self.obs_spectra.values )

    # #  =======  transform_PCA_Scaler_obs_spectra =====================================================
    # def transform_PCA_Scaler_obs_spectra(self):
    #     self.obs_spectra_scaled =  self.trained_StandardScaler_X.transform( self.obs_spectra.values )
    #     self.obs_spectra_scaled =  self.trained_pca_model.transform( self.obs_spectra_scaled )
    #
        
#     def transform_RFE_Scaler_obs_spectra(self):
# #         self.obs_spectra_featureDropped = self.obs_spectra.drop(columns=[str(x) for x in self.features2drop_df['features2drop']])
#         self.obs_spectra_featureskept = self.obs_spectra[self.features2keep_df['feature_names']]
#         self.obs_spectra_scaled =  self.trained_StandardScaler_X.transform( self.obs_spectra_featureskept.values )


    def predict(self):
        # class_ = self.trained_model.predict(self.obs_spectra_scaled)
        if self.ml_model_str != '1DCNN': 
            class_ = self.trained_model.predict(self.obs_spectra_scaled)[0]
        # elif self.ml_model_str == '1DCNN':
            # class_ = self.encoder.inverse_transform([np.argmax(class_)])[0]
        return class_
    
    
    
    def filter_bd_dataset(self, model_spectra, T,g,c_o,met):
        data = model_spectra[(model_spectra['temperature']==T) & 
                      (model_spectra['gravity']==g) & 
                      (model_spectra['c_o_ratio']==c_o) & 
                      (model_spectra['metallicity']==met)
                     ].reset_index().loc[0:1,'2.512':'0.897'].to_numpy().reshape(-1)
        return data
    
    def load_from_observational_spectra(self, observational_spectra):
        self.feature_names_obs = observational_spectra.feature_names_obs
        self.Fnu_obs_absolute = observational_spectra.Fnu_obs_absolute
        self.df_flux_object = observational_spectra.df_flux_object

        
    def plot_intervals(self,
                       x , y,
                      interv_percent, 
                      ):
        #create 95% confidence interval for population mean weight
        interval = st.t.interval(alpha=interv_percent, df=len(y)-1, loc=np.mean(y), scale=st.sem(y))
        y_up = [num+interval[0] for num in  y]
        y_low = [num-interval[1] for num in  y]
        return  y_low, y_up

    def benchmark_with_ML_derived_spectra(self, 
                                          observational_spectra_class_inst,
                                          browndwarf_df,
                                          interv_percent = 0.95):

        p = figure(width=800, height=400,
                  y_axis_type="linear",x_axis_type="linear",)
        
        
#         x= observational_spectra_class_inst.feature_names_obs
#         y= observational_spectra_class_inst.Fnu_obs_absolute
#         p.line(x,y, 
#                  color='green', legend_label = 'Fnu_obs_absolute')
#         y_low, y_up = self.plot_intervals(x, y, interv_percent)
#         p.varea(x=x, y1=y_low, y2=y_up,fill_color="green", alpha = 0.2)
        
        
        x = np.float32( observational_spectra_class_inst.df_flux_object.columns.values )
        y = observational_spectra_class_inst.df_flux_object.values[0]
        p.scatter(x, y, 
                  color='blue', 
                  legend_label = 'Fnu_obs_absolute_intd')
        y_low, y_up = self.plot_intervals(x, y, interv_percent)
        p.varea(x=x, y1=y_low, y2=y_up,fill_color="blue", alpha = 0.3)

        
        x = np.float32( observational_spectra_class_inst.df_flux_object.columns.values )
        y = self.filter_bd_dataset(browndwarf_df, 800, 4.75, 0.25, -0.5)
        p.scatter(x,y, 
                  color='red', 
                  legend_label = 'ML_predicted-Fnu')
        y_low, y_up = self.plot_intervals(x, y, interv_percent)
        p.varea(x=x, y1=y_low, y2=y_up,fill_color="red", alpha = 0.4)

        
        x = np.float32( observational_spectra_class_inst.df_flux_object.columns.values )
        y = self.filter_bd_dataset(browndwarf_df, 800, 4., 0.25, -0.3)
        p.scatter(x,y, 
                  color='black', 
                  legend_label = 'ZJ-derived values')
        y_low, y_up = self.plot_intervals(x, y, interv_percent)
        p.varea(x=x, y1=y_low, y2=y_up,fill_color="black", alpha = 0.3)

        
        p.xaxis.axis_label = 'Wavelength  [𝜇m]'
        p.yaxis.axis_label = 'Flux (Fν)'
        show(p)

        
    def benchmark_with_ML_derived_spectra_residual(self, 
                                          observational_spectra_class_inst,
                                          browndwarf_df,
                                          interv_percent = 0.95):

        p = figure(width=800, height=400,
                  y_axis_type="log",x_axis_type="linear",)
        
        
#         x= observational_spectra_class_inst.feature_names_obs
#         y= observational_spectra_class_inst.Fnu_obs_absolute
#         p.line(x,y, 
#                  color='green', legend_label = 'Fnu_obs_absolute')
#         y_low, y_up = self.plot_intervals(x, y, interv_percent)
#         p.varea(x=x, y1=y_low, y2=y_up,fill_color="green", alpha = 0.2)
        
        
        x_obs = np.float32( observational_spectra_class_inst.df_flux_object.columns.values )
        y_obs = observational_spectra_class_inst.df_flux_object.values[0]
        
        x = np.float32( observational_spectra_class_inst.df_flux_object.columns.values )
        y = self.filter_bd_dataset(browndwarf_df, 800, 4.75, 0.25, -0.5)
        p.scatter(x,abs(y-y_obs)*100/y_obs, 
                  color='red', 
                  legend_label = 'ML_predicted-Fnu')
#         y_low, y_up = self.plot_intervals(x-x_obs,y-y_obs, interv_percent)
#         p.varea(x=x, y1=y_low, y2=y_up,fill_color="red", alpha = 0.4)

        
        x = np.float32( observational_spectra_class_inst.df_flux_object.columns.values )
        y = self.filter_bd_dataset(browndwarf_df, 800, 4., 0.25, -0.3)
        p.scatter(x,abs(y-y_obs)*100/y_obs, 
                  color='black', 
                  legend_label = 'ZJ-derived values')
#         y_low, y_up = self.plot_intervals(x-x_obs,y-y_obs, interv_percent)
#         p.varea(x=x, y1=y_low, y2=y_up,fill_color="black", alpha = 0.3)

        
        p.xaxis.axis_label = 'Wavelength  [𝜇m]'
        p.yaxis.axis_label = 'Abs. Flux (Fν) Residual'
        show(p)      
        
        
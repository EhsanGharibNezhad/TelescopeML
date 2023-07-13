# Import functions from other modules ============================

from io_funs import LoadSave
# from functions import * #JUST CHANGED
from StatVisReportAnalyzer import plot_predicted_vs_observed, boxplot_hist, plot_spectra_errorbar, plot_predicted_vs_spectra_errorbar
from StatVisReportAnalyzer import filter_dataframe, interpolate_df, find_nearest_top_bottom, filter_dataset_range, regression_report, print_results_fun

# from train_ml_regression_2 import TrainMlRegression

# Import python libraries ========================================

# Dataset manipulation libraries
import pandas as pd
import numpy as np
import pickle as pk
from scipy import stats

import os
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.io import output_notebook

# ML algorithm libraries
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Data Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from uncertainties import ufloat

# from bokeh.io import output_notebook
# from bokeh.layouts import row, column
# output_notebook()
# from bokeh.plotting import show,figure
# TOOLTIPS = [
#     ("index", "$index"),
#     ("(x,y)", "($x, $y)"),
# ]
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



import astropy.units as u
from astropy.constants import c
from astropy.nddata import StdDevUncertainty, NDDataArray


from scipy.interpolate import interp1d
import os
import scipy.stats as st
from sklearn.preprocessing import StandardScaler




import random
from bokeh.palettes import Magma, Inferno, Plasma, Viridis, Cividis
from bokeh.palettes import  viridis, inferno

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
    def __init__(self, feature_values_obs, feature_values_obs_err, feature_names_obs, feature_names_synthetic):
        self.feature_values_obs = feature_values_obs
        self.feature_values_obs_err = feature_values_obs_err
        self.feature_names_obs = feature_names_obs
        self.feature_names_model = np.sort(feature_names_synthetic)
        self.Fnu_obs = self.convert_flux_density()[0]  # Convert Flam_obs to Fnu_obs
        self.Fnu_obs_err = self.convert_flux_density()[1]  # Convert Flam_obs to Fnu_obs

    def Flam2Fnu(self):
        """
        Convert F_lambda (ergs/s/cm²/Å) to F_nu (ergs/s/cm²/Hz)
        """
        fnu = (self.feature_values_obs * u.erg / u.s / u.Angstrom / u.cm ** 2).to(
            u.erg / u.s / u.cm ** 2 / u.Hz, equivalencies=u.spectral_density(self.feature_names_obs * u.micron))
        return fnu.to_value()

    def convert_flux_density(self):
        """
        Convert F_lambda to F_nu along with error propagation

        Parameters:
            flam_values (array-like): Array of flux density values in F_lambda
            flam_errors (array-like): Array of error bars for the flux density values in F_lambda
            wavelengths (array-like): Array of corresponding wavelengths in microns

        Returns:
            fnu_values (array): Array of flux density values in F_nu
            fnu_errors (array): Array of error bars for the flux density values in F_nu
        """
        # Warning!!! Double-check the Error converstion
        # Error prop: https://pythonhosted.org/uncertainties/

        flam_values = self.feature_values_obs
        flam_errors = self.feature_values_obs_err
        wavelengths = self.feature_names_obs

        xxx = NDDataArray(flam_values, uncertainty=StdDevUncertainty(flam_errors),
                 unit=(u.erg / u.s / u.Angstrom / u.cm**2))

        Fnu_all = xxx.convert_unit_to(unit=(u.erg / u.s / u.cm**2 / u.Hz), equivalencies = u.spectral_density( wavelengths * u.micron ) )


        # fnu = ( flam_values * u.erg / u.s / u.Angstrom / u.cm**2 ).to(u.erg / u.s / u.cm**2 / u.Hz, equivalencies = u.spectral_density( wavelengths * u.micron ) )

        # xx = [( ufloat(flam_values_,flam_errors_) * u.erg / u.s / u.Angstrom / u.cm**2 ).to(u.erg / u.s / u.cm**2 / u.Hz, equivalencies = u.spectral_density( wavelengths_ * u.micron ) ) for flam_values_,flam_errors_,wavelengths_ in zip( flam_values,flam_errors, wavelengths ) ]
        # print(xx)
              
              
        # Temporary
        #flam = flam_values * u.erg / u.s / u.Angstrom / u.cm ** 2
        #nu = (c / (wavelengths * u.micron)).to(u.Hz)
        #fnu = (flam * c / nu ** 2).to(u.erg / u.s / u.cm ** 2 / u.Hz)
        # fnu_values = fnu.to_value()
        fnu_values = Fnu_all.data

        # flam_errors = np.asarray(flam_errors)
        # flam_errors = flam_errors * u.erg / u.s / u.Angstrom / u.cm ** 2
        # fnu_errors = np.abs(fnu_values * np.sqrt((flam_errors / flam_values) ** 2))
        fnu_errors = Fnu_all.uncertainty.array
        
        return fnu_values, fnu_errors


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
        self.Fnu_obs_absolute_err = self.Fnu_obs_err * (
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



        
class ObsParameterPredictor:
    def __init__(self, 
                 object_name, 
                 dataset, 
                 wl, 
                 train_cnn_regression_class,
                ):
        self.object_name = object_name
        self.dataset = dataset
        self.wl = wl
        self.train_cnn_regression_class = train_cnn_regression_class
 


    def load_observational_spectra(self,
                                  plot_observational_spectra_errorbar_ = True):


        # load the observational spectra 
        obs_data = pd.read_csv(f'../../datasets/observational_spectra/{self.object_name}_fluxcal.dat', 
                           delim_whitespace=True, comment='#', names=('wl','F_lambda','F_lambda_error'), 
                           usecols=(0,1,2))#.dropna(inplace=True)

        # Clean the observational spectra: Replace negative fluxes with ZEROs and NAN to zeros
        obs_data['F_lambda']=obs_data['F_lambda'].mask(obs_data['F_lambda'].lt(0),0)
        obs_data['F_lambda'].replace(0, np.nan, inplace=True)

        # Interpolate the observational spectra 
        obs_data['F_lambda'].interpolate(inplace=True)
        
        self.obs_data = obs_data

    
        if plot_observational_spectra_errorbar_:
            plot_spectra_errorbar(object_name = self.object_name, 
                                  features = self.obs_data['wl'], 
                                  feature_values = self.obs_data['F_lambda'],
                                  error = self.obs_data['F_lambda_error'])    
    
    
    def Process_Observational_Dataset(self,
                                     print_results_ = True,
                                     plot_predicted_vs_observed_ = True,
                                     ):
        
        # Instintiate ProcessObservationalDataset class
        bd_object = ProcessObservationalDataset( feature_values_obs = self.obs_data['F_lambda'].values,
                                                feature_values_obs_err = self.obs_data['F_lambda_error'].values,
                                                 feature_names_obs  = self.obs_data['wl'].values, 
                                                 feature_names_synthetic = self.wl['wl'].values,
                                                 # feature_values_obs_error = fluxcal['F_lambda_error'].values,
                                        )




        # Add the BD derived values: name, Teff, logg, met, distance_pc, radius_Rjup
        if self.object_name == 'Ross458C':
            bd_object.bd_info('Ross458C','804','4.09','0.23', 11.509, 0.68 )
        if self.object_name == 'HD3651B':
            bd_object.bd_info('HD3651B','818','3.94','-0.22', 11.134, 0.81 )
        if self.object_name == 'GJ570D':
            bd_object.bd_info('GJ570D','818','3.94','-0.22', 5.884, 0.79 )    



        # Extract the original ML features from the obervational spectrum
        bd_object.flux_interpolated(print_results=False, 
                                    plot_results=True,
                                    use_spectres=True
                                   )
        bd_object.df_flux_object     


        # Extract the engeenered ML features from the obervational spectrum    
        bd_object.df_flux_object_min = bd_object.df_flux_object.min(axis=1)
        bd_object.df_flux_object_max = bd_object.df_flux_object.max(axis=1)

        df_MinMax_obs = pd.DataFrame(( bd_object.df_flux_object_min, bd_object.df_flux_object_max)).T


        XminXmax_Stand = self.train_cnn_regression_class.standardize_X_ColumnWise.transform(df_MinMax_obs.values)

        # X_Norm = (bd_object.df_flux_object.values[0] - bd_object.df_flux_object.min(axis=1)[0]) / (bd_object.df_flux_object.max(axis=1)[0] - bd_object.df_flux_object.min(axis=1)[0])
        # X_Norm = X_Norm * (1. - 0.) + 0.

        bd_mean = bd_object.df_flux_object.mean(axis=1)[0]  
        bd_std = bd_object.df_flux_object.std(axis=1)[0]     

        X_Scaled = (bd_object.df_flux_object.values[0] - bd_mean) / bd_std



        y_pred_train = np.array(self.train_cnn_regression_class.trained_model.predict([X_Scaled[::-1].reshape(1,104),XminXmax_Stand], verbose=0))[:,:,0].T
        y_pred_train_ = self.train_cnn_regression_class.standardize_y_ColumnWise.inverse_transform( y_pred_train )
        y_pred_train_ [:,3] = 10**y_pred_train_[:,3]
        y_pred = y_pred_train_
        self.y_pred = y_pred
        self.targets_dic = dict ( zip( ['gravity', 'c_o_ratio', 'metallicity', 'temperature'] , [self.y_pred[0][i] for i in range(4)]) )
        self.bd_object = bd_object
    
    
        # ********************************* 

        if print_results_:
            print_results_fun(targets = self.targets_dic, 
                              print_title= 'Predicted Targets:')
    
        if plot_predicted_vs_observed_: 
            plot_predicted_vs_observed(training_datasets = self.dataset, 
                                        wl = self.wl,
                                        predicted_targets_dic = self.targets_dic,
                                        object_name = self.object_name,
                                        bd_object_class = bd_object,
                                      )


        # ********************************* 

        
    def predict_from_random_spectra(self,
                                   random_spectra_num = 10,
                                   print_results_ = True,
                                   plot_randomly_generated_spectra__ = True,
                                   plot_histogram_ = True,
                                   plot_boxplot_hist_ = True,
                                    plot_predicted_vs_observed_ = True,
                                    plot_predicted_vs_spectra_errorbar_ = True,
                                    plot_predictedRandomSpectra_vs_ObservedSpectra_errorbar_ = True,
                                    calculate_confidence_intervals_std_df_ = True,
                                   ):
        
        color = viridis(250).__iter__()
        # if random_spectra_num < 101:
            # color = viridis(random_spectra_num).__iter__()

        spectra_list = []
        param_list = []    

        for i in range(random_spectra_num):
            # Comment from Natasha: 
            spectra = pd.DataFrame( np.random.normal(self.obs_data['F_lambda'] , self.obs_data['F_lambda_error'] ) ,
                                     columns=['F_lambda'])

            # Process the dataset
            spectra['F_lambda'] = spectra['F_lambda'].mask(spectra['F_lambda'].lt(0),0)
            spectra['F_lambda'].replace(0, np.nan, inplace=True)
            spectra['F_lambda'].interpolate(inplace=True)
            
            # warning!!! check with Natasha
            # spectra['F_lambda_error'] = spectra['F_lambda_error'].mask(spectra['F_lambda_error'].lt(0),np.mean(spectra['F_lambda_error']))
            # spectra['F_lambda_error'].replace(np.mean(spectra['F_lambda_error']), np.nan, inplace=True)
            # spectra['F_lambda_error'].interpolate(inplace=True)


            # Instintiate ProcessObservationalDataset class
            bd_object_generated = ProcessObservationalDataset( feature_values_obs = spectra['F_lambda'].values,
                                                    feature_values_obs_err = self.obs_data['F_lambda_error'].values,
                                                     feature_names_obs  = self.obs_data['wl'].values, 
                                                     feature_names_synthetic = self.wl['wl'].values,
                                                     # feature_values_obs_error = fluxcal['F_lambda_error'].values,
                                            )
            self.bd_object_generated = bd_object_generated




            # Add the BD derived values: name, Teff, logg, met, distance_pc, radius_Rjup
            if self.object_name == 'Ross458C':
                bd_object_generated.bd_info('Ross458C','804','4.09','0.23', 11.509, 0.68 )
            if self.object_name == 'HD3651B':
                bd_object_generated.bd_info('HD3651B','818','3.94','-0.22', 11.134, 0.81 )
            if self.object_name == 'GJ570D':
                bd_object_generated.bd_info('GJ570D','818','3.94','-0.22', 5.884, 0.79 )    



            # Extract the original ML features from the obervational spectrum
            bd_object_generated.flux_interpolated(print_results= False, 
                                        plot_results= False,
                                        use_spectres=True
                                       )
            bd_object_generated.df_flux_object     


            # Extract the engeenered ML features from the obervational spectrum    
            bd_object_generated.df_flux_object_min = bd_object_generated.df_flux_object.min(axis=1)
            bd_object_generated.df_flux_object_max = bd_object_generated.df_flux_object.max(axis=1)

            df_MinMax_obs = pd.DataFrame(( bd_object_generated.df_flux_object_min, bd_object_generated.df_flux_object_max)).T

            XminXmax_Stand = self.train_cnn_regression_class.standardize_X_ColumnWise.transform(df_MinMax_obs.values)

            bd_mean = bd_object_generated.df_flux_object.mean(axis=1)[0]  
            bd_std = bd_object_generated.df_flux_object.std(axis=1)[0]     

            X_Scaled = (bd_object_generated.df_flux_object.values[0] - bd_mean) / bd_std


            y_pred_train = np.array(self.train_cnn_regression_class.trained_model.predict([X_Scaled[::-1].reshape(1,104),XminXmax_Stand], verbose=0))[:,:,0].T
            y_pred_train_ = self.train_cnn_regression_class.standardize_y_ColumnWise.inverse_transform( y_pred_train )
            y_pred_train_ [:,3] = 10**y_pred_train_[:,3]
            y_pred = y_pred_train_
            self.y_pred_random = y_pred
            self.targets_dic_random = dict ( zip( ['gravity', 'c_o_ratio', 'metallicity', 'temperature'] , [self.y_pred[0][i] for i in range(4)]) )

            spectra_list.append(bd_object_generated.df_flux_object.values)
            param_list.append(  self.y_pred_random[0] )
            
        self.spectra_list = spectra_list
        self.param_list = param_list

            # ********************************* 

        self.df_random_pred = pd.DataFrame(self.param_list, columns=['logg' ,'c_o' ,'met' ,'T'] )

        if print_results_:
            print_results_fun(targets = self.targets_dic_random, 
                              print_title= 'Predicted Targets:')

            display(self.df_random_pred.describe())



        if plot_randomly_generated_spectra__:
            p = figure(title=self.object_name+": Randomly generated spectra within 1σ", 
                       x_axis_label='Features (Wavelength [𝜇m])', y_axis_label='Flux (Fν)',
                       width=1000, height=300,
                       y_axis_type="log", background_fill_color="#fafafa"
                      )
            
            if random_spectra_num > 250:
                 random_spectra_num_index = 200
            else:
                 random_spectra_num_index = random_spectra_num
                
            for i in range(0,random_spectra_num_index):
                    p.line(self.wl.wl.values[::-1],self.spectra_list[i][0], 
                           line_width = 1,
                           line_alpha = 0.6, 
                           line_color=next(color),
                           )

                    
                    
            # Increase size of x and y ticks
            p.title.text_font_size = '12pt'
            p.xaxis.major_label_text_font_size = '12pt'
            p.xaxis.axis_label_text_font_size = '12pt'
            p.yaxis.major_label_text_font_size = '12pt'
            p.yaxis.axis_label_text_font_size = '12pt'


            # p.legend.location = "top_right"
            # p.legend.background_fill_color = 'white'
            # p.legend.background_fill_alpha = 0.5

            show(p)

            

            
            
        if plot_histogram_:
            plt.figure()
            self.df_random_pred.hist()
            plt.show()
        
        if plot_boxplot_hist_: 
            boxplot_hist(self.df_random_pred['logg'],  x_label=r'$\log g$', xy_loc=[0.05,0.98],)
            boxplot_hist(self.df_random_pred['T'],     x_label=r'$T_{eff}$', xy_loc=[0.05,0.98],)
            boxplot_hist(self.df_random_pred['c_o'],   x_label=r'C/O', xy_loc=[0.05,0.98],)
            boxplot_hist(self.df_random_pred['met'],   x_label=r'[M/H]', xy_loc=[0.05,0.98],)
            

        if plot_predicted_vs_observed_: 
            targets = dict ( zip( ['gravity', 'c_o_ratio', 'metallicity', 'temperature'] ,  list( self.df_random_pred.agg(np.mean) ) ) )
            plot_predicted_vs_observed(training_datasets = self.dataset, 
                                        wl = self.wl,
                                        predicted_targets_dic = targets,
                                        object_name = self.object_name,
                                        bd_object_class = self.bd_object,
                                        print_results=True, 
                                      )

        if plot_predicted_vs_spectra_errorbar_: 
            targets = dict ( zip( ['gravity', 'c_o_ratio', 'metallicity', 'temperature'] ,  list( self.df_random_pred.agg(np.mean) ) ) )
            plot_predicted_vs_spectra_errorbar(
                                  object_name = self.object_name, 
                                  features = self.obs_data['wl'], 
                                  feature_values = self.bd_object.Fnu_obs_absolute,
                                  error = np.array(self.bd_object.Fnu_obs_absolute_err),
                              training_datasets = self.dataset,  
                               wl = self.wl,
                               predicted_targets_dic = targets,
                               bd_object_class = self.bd_object,
                               print_results_ = True)
            
        if calculate_confidence_intervals_std_df_:
            self.confidence_intervals_std_df = calculate_confidence_intervals_std_df ( 
                                dataset= pd.DataFrame (data= np.array(self.spectra_list).reshape(-1,104) , columns=self.wl.wl),
                                        _print_results_= True,
                                        plot_calculate_confidence_intervals_std_df= True,
                                      )
            
        if plot_predictedRandomSpectra_vs_ObservedSpectra_errorbar_:
            targets = dict ( zip( ['gravity', 'c_o_ratio', 'metallicity', 'temperature'] ,  list( self.df_random_pred.agg(np.mean) ) ) )
            plot_predictedRandomSpectra_vs_ObservedSpectra_errorbar(
                                  stat_df = self.confidence_intervals_std_df,
                                    confidence_level = .95,
                                              object_name = self.object_name, 
                                  features = self.obs_data['wl'], 
                                  feature_values = self.bd_object.Fnu_obs_absolute,
                                  error = np.array(self.bd_object.Fnu_obs_absolute_err),
                              training_datasets = self.dataset,  
                               wl = self.wl,
                               predicted_targets_dic = targets,
                               bd_object_class = self.bd_object,
                               print_results_ = True,)
                

def calculate_confidence_intervals_std_df (dataset,
                                           _print_results_ = False,
                                           plot_calculate_confidence_intervals_std_df = False):
    df3 = dataset#[::-1]#.iloc[1:,4:-1]

    confidence_level = 0.95  # Confidence level (e.g., 95% confidence)

    # Calculate the sample size
    n = len(df3)

    # Calculate the mean and standard deviation for each column
    mean_values = df3.mean()
    std_values = df3.std()

    # print(mean_values, std_values)
    # Calculate the standard error
    se_values = std_values / np.sqrt(n)

    # Calculate the t-value for the desired confidence level and degrees of freedom
    t_value = stats.t.ppf((1 + confidence_level) / 2, df=n - 1)

    # Calculate the confidence interval for each column
    stat_df = pd.DataFrame(columns=['confidence_level_lower', 'confidence_level_upper'], index=None)
    stat_df = stat_df.astype(np.float128)

    for column in df3.columns:
        lower_bound = mean_values[column] - t_value * se_values[column]
        upper_bound = mean_values[column] + t_value * se_values[column]
        stat_df.loc[column] = [lower_bound, upper_bound]
        
    stat_df['mean'] = mean_values
    stat_df['std_values'] = std_values
    stat_df['wl'] = np.float64(df3.columns)
    # stat_df['mean-std'] = mean_values - std_values
    # stat_df['mean+std'] = mean_values + std_values
    
    if _print_results_:
        display(stat_df)
    
    
    if plot_calculate_confidence_intervals_std_df: 
        # Plot the mean and confidence intervals
        # fig, ax = plt.subplots(figsize=(14,4))
        x = np.round(df3.columns,2)#[float(s) for s in list(df3.columns)] #np.arange(len(df3.columns))
        # ax.set_xticks(size=6)
        # ax.set_xticklabels( rotation=45, size=6)

        # Create a figure
        p = figure(title='Mean with Confidence Intervals', 
                   x_axis_label='Features (Wavelength [μm])',
                   y_axis_label='Flux (Fν)', 
                   y_axis_type="log",
                   width=1000, height=400)
        


        # Plot the mean as a line
        p.line(x = stat_df['wl'][::-1], y = stat_df['mean'], color='blue', line_width=2, legend_label='Mean')

        # Plot the shaded regions for confidence intervals
        p.varea(x = stat_df['wl'][::-1], y1=stat_df['confidence_level_lower'], y2=stat_df['confidence_level_upper'], fill_color='red', fill_alpha=0.8,
                legend_label='Confidence Level: {}%'.format(confidence_level))

        # Plot the shaded regions for 1 sigma
        p.varea(x = stat_df['wl'][::-1], y1=stat_df['mean']-stat_df['std_values'], y2=stat_df['mean']+stat_df['std_values'], fill_color='green', fill_alpha=0.4,
                legend_label='1σ')

        # Customize the plot
        # p.legend.location = 'top_left'
        p.legend.click_policy = 'hide'

        # Show the plot
        show(p)


        
    return stat_df





def plot_predictedRandomSpectra_vs_ObservedSpectra_errorbar (stat_df,
                                                             confidence_level,
                                                            object_name, 
                                                              features, 
                                                              feature_values, 
                                                              error, 
                                                              training_datasets, 
                                                                   wl,
                                                                   predicted_targets_dic,
                                                                   bd_object_class,
                                                                   print_results_ = True,):



        # Create a figure
        p = figure(title='Mean with Confidence Intervals', 
                   x_axis_label='Features (Wavelength [μm])',
                   y_axis_label='Flux (Fν)', 
                   y_axis_type="log",
                   width=1000, height=400)
        


        # Plot the mean as a line
        p.line(x = stat_df['wl'][::-1], y = stat_df['mean'], color='blue', line_width=2, legend_label='Mean')

        # Plot the shaded regions for confidence intervals
        p.varea(x = stat_df['wl'][::-1], y1=stat_df['confidence_level_lower'], y2=stat_df['confidence_level_upper'], fill_color='red', fill_alpha=0.8,
                legend_label='Confidence Level: {}%'.format(confidence_level))

        # Plot the shaded regions for 1 sigma
        p.varea(x = stat_df['wl'][::-1], y1=stat_df['mean']-stat_df['std_values'], y2=stat_df['mean']+stat_df['std_values'], fill_color='green', fill_alpha=0.4,
                legend_label='1σ')
        
        
        # ++++++++++++++++++++++++++++++++
        
            # Calculate the error bar coordinates
        upper = [y_val + err_val for y_val, err_val in zip(feature_values, error)]
        lower = [y_val - err_val for y_val, err_val in zip(feature_values, error)]

        # Create a ColumnDataSource to store the data
        source = ColumnDataSource(data=dict(x=features, y=feature_values, upper=upper, lower=lower))
        
        # Add the scatter plot
        p.scatter('x', 'y', source=source, size=4, fill_color='green', line_color=None, line_alpha=0.2, legend_label=f"{object_name}: Observational data")

        # Add the error bars using segment
        p.segment(x0='x', y0='lower', x1='x', y1='upper', source=source, color='gray', line_alpha=0.7)

    
        # Customize the plot
        # p.legend.location = 'top_left'
        # Increase size of x and y ticks
        p.title.text_font_size = '12pt'
        p.xaxis.major_label_text_font_size = '12pt'
        p.xaxis.axis_label_text_font_size = '12pt'
        p.yaxis.major_label_text_font_size = '12pt'
        p.yaxis.axis_label_text_font_size = '12pt'


        p.legend.location = "bottom_left"
        p.legend.background_fill_color = 'white'
        p.legend.background_fill_alpha = 0.5
        p.legend.click_policy = 'hide'

        # Show the plot
        show(p)



        
def plot_predicted_vs_spectra_errorbar(
                          object_name, 
                          features, 
                          feature_values, 
                          error, 
                          training_datasets, 
                               wl,
                               predicted_targets_dic,
                               bd_object_class,
                               print_results_ = True,
                            ):
    
    # Calculate the error bar coordinates
    upper = [y_val + err_val for y_val, err_val in zip(feature_values, error)]
    lower = [y_val - err_val for y_val, err_val in zip(feature_values, error)]

    # Create a ColumnDataSource to store the data
    source = ColumnDataSource(data=dict(x=features, y=feature_values, upper=upper, lower=lower))

    # Create the Observational figure ***********************************
    p = figure(title=f"{object_name}: Calibrated Observational Spectra",
               x_axis_label="Features (Wavelength [𝜇m])",
               y_axis_label="Flux (F𝜈)",
               width=1000, height=300,
               y_axis_type="log",
               tools="pan,wheel_zoom,box_zoom,reset")



    # Add the scatter plot
    p.scatter('x', 'y', source=source, size=4, fill_color='green', line_color=None, line_alpha=0.2, legend_label=f"{object_name}: Observational data")

    # Add the error bars using segment
    p.segment(x0='x', y0='lower', x1='x', y1='upper', source=source, color='gray', line_alpha=0.7)


    
    # Create the Predicted figure ***********************************
    
    ypred = list( predicted_targets_dic.values() )
    
    filtered_df = interpolate_df(dataset=training_datasets, 
                       predicted_targets_dic = predicted_targets_dic,
                       print_results_ = False)

    display(filtered_df)
    
    # Add the scatter plot

    p.line(x =wl['wl'] , y=filtered_df.drop(columns=['gravity', 'c_o_ratio', 'metallicity', 'temperature','is_augmented']).values[0], 
           line_width = 1,
           legend_label= 'ML Predicted:'+', '.join([['log𝑔= ','C/O= ', '[M/H]= ', 'T= '][i]+str(np.round(ypred[i],2)) for i in  range(4)]))

    # Increase size of x and y ticks
    p.title.text_font_size = '12pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.xaxis.axis_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'
    p.yaxis.axis_label_text_font_size = '12pt'


    p.legend.location = "bottom_left"
    p.legend.background_fill_color = 'white'
    p.legend.background_fill_alpha = 0.5


    if print_results_:
        display(bd_object_class.df_flux_object.iloc[:, ::-1])


    # Show the plot
    output_notebook()
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
        
        
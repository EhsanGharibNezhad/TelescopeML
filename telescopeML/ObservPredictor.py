# Import functions from other modules ============================

from io_funs import LoadSave
# from functions import * #JUST CHANGED
from StatVisAnalyzer import plot_predicted_vs_observed, boxplot_hist, plot_spectra_errorbar, plot_predicted_vs_spectra_errorbar
from StatVisAnalyzer import filter_dataframe, interpolate_df, find_nearest_top_bottom, filter_dataset_range, regression_report, print_results_fun

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
from sklearn.preprocessing import StandardScaler


# Data Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from uncertainties import ufloat

from skopt.plots import plot_evaluations


# from bokeh.palettes import Category20, colorblind
from tensorflow.keras.models import Sequential, model_from_json
#



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
    - feature_values_obs_err: 
    - feature_names_obs: Name of features (wavelength) from observational data, e.g., 0.9, 1.0, 1.1 micron
    - feature_names_synthetic: Name of features (wavelengths) from synthetic data
    """
    def __init__(self, 
                 feature_values_obs, 
                 feature_values_obs_err, 
                 feature_names_obs, 
                 feature_names_synthetic,
                 bd_literature_dic = None
                ):
        
        self.feature_values_obs = feature_values_obs
        self.feature_values_obs_err = feature_values_obs_err
        self.feature_names_obs = feature_names_obs
        self.feature_names_model = np.sort(feature_names_synthetic)
        self.Fnu_obs = self.Flam2Fnu()[0]  # Convert Flam_obs to Fnu_obs
        self.Fnu_obs_err = self.Flam2Fnu()[1]  # Convert Flam_obs to Fnu_obs
        
        if bd_literature_dic['bd_distance_pc'] != None and bd_literature_dic['bd_radius_Rjup'] != None:
            self.Fnu_obs_absolute = self.Fnu_obs * (
                    bd_literature_dic['bd_distance_pc'] * ((u.pc).to(u.jupiterRad)) / (bd_literature_dic['bd_radius_Rjup'])) ** 2
            self.Fnu_obs_absolute_err = self.Fnu_obs_err * (
                    bd_literature_dic['bd_distance_pc'] * ((u.pc).to(u.jupiterRad)) / (bd_literature_dic['bd_radius_Rjup'])) ** 2
        
    
    def Flam2Fnu(self):
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

        flam_dataset = NDDataArray(flam_values, uncertainty=StdDevUncertainty(flam_errors),
                 unit=(u.erg / u.s / u.Angstrom / u.cm**2))

        Fnu_all = flam_dataset.convert_unit_to(unit=(u.erg / u.s / u.cm**2 / u.Hz), equivalencies = u.spectral_density( wavelengths * u.micron ) )
        fnu_values = Fnu_all.data
        fnu_errors = Fnu_all.uncertainty.array
        
        return fnu_values, fnu_errors


        
        
        
    def flux_interpolated(self, print_results=False, plot_results=True, use_spectres=True):
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
        p = figure(width=800, height=300, y_axis_type="log", x_axis_type="linear")

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



        
class ObsParameterPredictor_oldVersion:
    def __init__(self, 
                 object_name, 
                 training_dataset_df, 
                 wl, 
                 train_cnn_regression_class,
                 bd_literature_dic,

                 
                ):
        self.object_name = object_name
        self.training_dataset_df = training_dataset_df
        self.wl = wl
        self.train_cnn_regression_class = train_cnn_regression_class
        self.bd_literature_dic = bd_literature_dic
 #         Parameters
#         -----------
#         - bd_name: Brown dwarf name
#         - bd_Teff: Brown dwarf effective temperature
#         - bd_logg: Brown dwarf log(surface gravity) in cgs
#         - bd_met: Brown dwarf metallicity
#         - bd_distance_pc: Distance of the object from Earth in pc unit
#         - bd_radius_Rjup: Radius of the object in Jupiter radius

#         Returns
#         --------
#         - Calculate Fnu_obs_absolute from Fnu_obs
#         """
#         self.bd = {}
#         self.bd['bd_name'] = bd_name
#         self.bd['bd_Teff'] = bd_Teff
#         self.bd['bd_logg'] = bd_logg
#         self.bd['bd_met'] = bd_met
        

    def load_observational_spectra(self,
                                  plot_observational_spectra_errorbar_ = True,
                                  _replace_zeros_with_mean_ = True,
                                  ):


        # load the observational spectra 
        obs_data_df = pd.read_csv(f'../../datasets/observational_spectra/{self.object_name}_fluxcal.dat', 
                           delim_whitespace=True, comment='#', names=('wl','F_lambda','F_lambda_error'), 
                           usecols=(0,1,2))#.dropna(inplace=True)
            

        # Process the dataset
        obs_data_df['F_lambda'] = obs_data_df['F_lambda'].mask(obs_data_df['F_lambda'].lt(0),0)
        obs_data_df['F_lambda'].replace(0, np.nan, inplace=True)
        obs_data_df['F_lambda'].interpolate(inplace=True)


        
        if _replace_zeros_with_mean_:
            # obs_data_df['F_lambda'] = replace_zeros_with_mean(obs_data_df['F_lambda'])
            obs_data_df['F_lambda_error'] = replace_zeros_with_mean(obs_data_df['F_lambda_error'])
            #     obs_data_df['F_lambda'] = replace_zeros_with_mean(obs_data_df['F_lambda'])

        


        
        self.obs_data_df = obs_data_df
        display(self.obs_data_df)

    
        if plot_observational_spectra_errorbar_:
            plot_spectra_errorbar(object_name = self.object_name, 
                                  features = self.obs_data_df['wl'], 
                                  feature_values = self.obs_data_df['F_lambda'],
                                  error = self.obs_data_df['F_lambda_error'])    
    
    
    def Process_Observational_Dataset(self,
                                     print_results_ = True,
                                     plot_predicted_vs_observed_ = True,
                                     ):
        
        # Instintiate ProcessObservationalDataset class
        bd_object = ProcessObservationalDataset( feature_values_obs = self.obs_data_df['F_lambda'].values,
                                                feature_values_obs_err = self.obs_data_df['F_lambda_error'].values,
                                                 feature_names_obs  = self.obs_data_df['wl'].values, 
                                                 feature_names_synthetic = self.wl['wl'].values,
                                                 bd_literature_dic = self.bd_literature_dic,
                                        )



        # Extract the original ML features from the obervational spectrum
        bd_object.flux_interpolated(print_results=False, 
                                    plot_results=True,
                                    use_spectres=True
                                   )
        if print_results_:
            print('------------ df_flux_object ------------')
            display ( bd_object.df_flux_object )  


        # Extract the engeenered ML features from the obervational spectrum    
        bd_object.df_flux_object_min = bd_object.df_flux_object.min(axis=1)
        bd_object.df_flux_object_max = bd_object.df_flux_object.max(axis=1)

        df_MinMax_obs = pd.DataFrame(( bd_object.df_flux_object_min, bd_object.df_flux_object_max)).T

        if print_results_:
            print('------------ df_MinMax_obs ------------')
            display(df_MinMax_obs)


        XminXmax_Stand = self.train_cnn_regression_class.standardize_X_ColumnWise.transform(df_MinMax_obs.values)

        # X_Norm = (bd_object.df_flux_object.values[0] - bd_object.df_flux_object.min(axis=1)[0]) / (bd_object.df_flux_object.max(axis=1)[0] - bd_object.df_flux_object.min(axis=1)[0])
        # X_Norm = X_Norm * (1. - 0.) + 0.

        bd_mean = bd_object.df_flux_object.mean(axis=1)[0]  
        bd_std = bd_object.df_flux_object.std(axis=1)[0]    
        print(bd_literature_dic['bd_radius_Rjup'], bd_mean, bd_std)

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
            plot_predicted_vs_observed(training_datasets = self.training_dataset_df, 
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
            spectra = pd.DataFrame( np.random.normal(self.obs_data_df['F_lambda'] , self.obs_data_df['F_lambda_error'] ) ,
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
                                                    feature_values_obs_err = self.obs_data_df['F_lambda_error'].values,
                                                     feature_names_obs  = self.obs_data_df['wl'].values, 
                                                     feature_names_synthetic = self.wl['wl'].values,
                                                      bd_literature_dic = self.bd_literature_dic,
                                                     # feature_values_obs_error = fluxcal['F_lambda_error'].values,
                                            )
            self.bd_object_generated = bd_object_generated







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
            plot_predicted_vs_observed(training_datasets = self.training_dataset_df, 
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
                                  features = self.obs_data_df['wl'], 
                                  feature_values = self.bd_object.Fnu_obs_absolute,
                                  error = np.array(self.bd_object.Fnu_obs_absolute_err),
                              training_datasets = self.training_dataset_df,  
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
                                        features = self.obs_data_df['wl'], 
                                        feature_values = self.bd_object.Fnu_obs_absolute,
                                        error = np.array(self.bd_object.Fnu_obs_absolute_err),
                                        training_datasets = self.training_dataset_df,  
                                        wl = self.wl,
                                        predicted_targets_dic = targets,
                                        bd_object_class = self.bd_object,
                                        print_results_ = True,
                                        )
                
                
                

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
#         upper = [y_val + err_val for y_val, err_val in zip(feature_values, error)]
#         lower = [y_val - err_val for y_val, err_val in zip(feature_values, error)]

#         # Add a small offset to the feature values
#         offset = 2*np.mean(error)
#         feature_values_offset = [val if val <= 0 else val + offset for val in feature_values]
#         upper_offset = [val + offset for val in upper]
#         lower_offset = [val + offset for val in lower]

#         # Create a ColumnDataSource to store the data
#         source = ColumnDataSource(data=dict(x=features, y=feature_values_offset, upper=upper_offset, lower=lower_offset))        
#             # Calculate the error bar coordinates
        upper = [abs(y_val + err_val) for y_val, err_val in zip(feature_values, error)]
        # lower = [y_val - err_val for y_val, err_val in zip(feature_values, error)]
        lower = [abs(y_val - err_val) for y_val, err_val in zip(feature_values, error)]


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
    #     upper = [y_val + err_val for y_val, err_val in zip(feature_values, error)]
    #     lower = [y_val - err_val for y_val, err_val in zip(feature_values, error)]

    #     # Add a small offset to the feature values
    #     offset = 2*np.mean(error)
    #     feature_values_offset = [val if val <= 0 else val + offset for val in feature_values]
    #     upper_offset = [val + offset for val in upper]
    #     lower_offset = [val + offset for val in lower]

    # Create a ColumnDataSource to store the data
    # source = ColumnDataSource(data=dict(x=features, y=feature_values_offset, upper=upper_offset, lower=lower_offset))      

    upper = [abs(y_val + err_val) for y_val, err_val in zip(feature_values, error)]
    #lower = [y_val - err_val for y_val, err_val in zip(feature_values, error)]
    lower = [abs(y_val - err_val) for y_val, err_val in zip(feature_values, error)]

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
 
        
        
def replace_zeros_with_mean(df_col):
    # Ref: ChatGPT
    # Replace zero values with the mean of their non-zero neighbors
    zero_indices = np.where(df_col.values <= 0)

    if zero_indices[0].size > 0:
        for row in zero_indices[0]:
            neighbors = df_col.iloc[max(0, row-1):row+2]
            neighbors = neighbors[neighbors != 0]
            while len(neighbors) == 0:
                row -= 1
                neighbors = df_col.iloc[max(0, row-1):row+2]
                neighbors = neighbors[neighbors != 0]
            df_col.iloc[row] = np.mean(neighbors)

        return df_col

    else:
        print("No zero values found in the column.")
        
  


# ===================================================================


       
class ObsParameterPredictor:
    def __init__(self, 
                 object_name, 
                 training_dataset_df, 
                 wl, 
                 train_cnn_regression_class,
                 bd_literature_dic,

                 
                ):
        self.object_name = object_name
        self.training_dataset_df = training_dataset_df
        self.wl = wl
        self.train_cnn_regression_class = train_cnn_regression_class
        self.bd_literature_dic = bd_literature_dic
 #         Parameters
#         -----------
#         - bd_name: Brown dwarf name
#         - bd_Teff: Brown dwarf effective temperature
#         - bd_logg: Brown dwarf log(surface gravity) in cgs
#         - bd_met: Brown dwarf metallicity
#         - bd_distance_pc: Distance of the object from Earth in pc unit
#         - bd_radius_Rjup: Radius of the object in Jupiter radius


# class ProcessObservationalDataset:
    """
    Load, process, visualize observational spectra

    Attribute
    ---------
    - feature_values_obs: Fluxes for each feature (wavelength) from observational data
    - feature_values_obs_err: 
    - feature_names_obs: Name of features (wavelength) from observational data, e.g., 0.9, 1.0, 1.1 micron
    - feature_names_synthetic: Name of features (wavelengths) from synthetic data
    """
    def ProcessObservationalDataset(self, 
                 feature_values_obs, 
                 feature_values_obs_err, 
                 feature_names_obs, 
                 feature_names_synthetic,
                 bd_literature_dic = None
                ):
        
        self.feature_values_obs = feature_values_obs
        self.feature_values_obs_err = feature_values_obs_err
        self.feature_names_obs = feature_names_obs
        self.feature_names_model = np.sort(feature_names_synthetic)
        self.Fnu_obs = self.Flam2Fnu()[0]  # Convert Flam_obs to Fnu_obs
        self.Fnu_obs_err = self.Flam2Fnu()[1]  # Convert Flam_obs to Fnu_obs
        
        if bd_literature_dic['bd_distance_pc'] != None and bd_literature_dic['bd_radius_Rjup'] != None:
            self.Fnu_obs_absolute = self.Fnu_obs * (
                    bd_literature_dic['bd_distance_pc'] * ((u.pc).to(u.jupiterRad)) / (bd_literature_dic['bd_radius_Rjup'])) ** 2
            self.Fnu_obs_absolute_err = self.Fnu_obs_err * (
                    bd_literature_dic['bd_distance_pc'] * ((u.pc).to(u.jupiterRad)) / (bd_literature_dic['bd_radius_Rjup'])) ** 2
        
    
    def Flam2Fnu(self):
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

        flam_dataset = NDDataArray(flam_values, uncertainty=StdDevUncertainty(flam_errors),
                 unit=(u.erg / u.s / u.Angstrom / u.cm**2))

        Fnu_all = flam_dataset.convert_unit_to(unit=(u.erg / u.s / u.cm**2 / u.Hz), equivalencies = u.spectral_density( wavelengths * u.micron ) )
        fnu_values = Fnu_all.data
        fnu_errors = Fnu_all.uncertainty.array
        
        return fnu_values, fnu_errors


        
        
        
    def flux_interpolated(self, print_results=False, plot_results=True, use_spectres=True):
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
        p = figure(width=800, height=300, y_axis_type="log", x_axis_type="linear")

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


    def load_observational_spectra(self,
                                  plot_observational_spectra_errorbar_ = True,
                                  _replace_zeros_with_mean_ = True,
                                  ):


        # load the observational spectra 
        obs_data_df = pd.read_csv(f'../../datasets/observational_spectra/{self.object_name}_fluxcal.dat', 
                           delim_whitespace=True, comment='#', names=('wl','F_lambda','F_lambda_error'), 
                           usecols=(0,1,2))#.dropna(inplace=True)
            

        # Process the dataset
        obs_data_df['F_lambda'] = obs_data_df['F_lambda'].mask(obs_data_df['F_lambda'].lt(0),0)
        obs_data_df['F_lambda'].replace(0, np.nan, inplace=True)
        obs_data_df['F_lambda'].interpolate(inplace=True)


        
        if _replace_zeros_with_mean_:
            # obs_data_df['F_lambda'] = replace_zeros_with_mean(obs_data_df['F_lambda'])
            obs_data_df['F_lambda_error'] = replace_zeros_with_mean(obs_data_df['F_lambda_error'])
            #     obs_data_df['F_lambda'] = replace_zeros_with_mean(obs_data_df['F_lambda'])

        


        
        self.obs_data_df = obs_data_df
        display(self.obs_data_df)

    
        if plot_observational_spectra_errorbar_:
            plot_spectra_errorbar(object_name = self.object_name, 
                                  features = self.obs_data_df['wl'], 
                                  feature_values = self.obs_data_df['F_lambda'],
                                  error = self.obs_data_df['F_lambda_error'])    
    
    
    def Process_Observational_Dataset(self,
                                     print_results_ = True,
                                     plot_predicted_vs_observed_ = True,
                                     ):
        
        # Instintiate ProcessObservationalDataset class
        bd_object = ProcessObservationalDataset( feature_values_obs = self.obs_data_df['F_lambda'].values,
                                                feature_values_obs_err = self.obs_data_df['F_lambda_error'].values,
                                                 feature_names_obs  = self.obs_data_df['wl'].values, 
                                                 feature_names_synthetic = self.wl['wl'].values,
                                                 bd_literature_dic = self.bd_literature_dic,
                                        )



        # Extract the original ML features from the obervational spectrum
        bd_object.flux_interpolated(print_results=False, 
                                    plot_results=True,
                                    use_spectres=True
                                   )
        if print_results_:
            print('------------ df_flux_object ------------')
            display ( bd_object.df_flux_object )  


        # Extract the engeenered ML features from the obervational spectrum    
        bd_object.df_flux_object_min = bd_object.df_flux_object.min(axis=1)
        bd_object.df_flux_object_max = bd_object.df_flux_object.max(axis=1)

        df_MinMax_obs = pd.DataFrame(( bd_object.df_flux_object_min, bd_object.df_flux_object_max)).T

        if print_results_:
            print('------------ df_MinMax_obs ------------')
            display(df_MinMax_obs)


        XminXmax_Stand = self.train_cnn_regression_class.standardize_X_ColumnWise.transform(df_MinMax_obs.values)

        # X_Norm = (bd_object.df_flux_object.values[0] - bd_object.df_flux_object.min(axis=1)[0]) / (bd_object.df_flux_object.max(axis=1)[0] - bd_object.df_flux_object.min(axis=1)[0])
        # X_Norm = X_Norm * (1. - 0.) + 0.

        bd_mean = bd_object.df_flux_object.mean(axis=1)[0]  
        bd_std = bd_object.df_flux_object.std(axis=1)[0]    
        print(bd_literature_dic['bd_radius_Rjup'], bd_mean, bd_std)

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
            plot_predicted_vs_observed(training_datasets = self.training_dataset_df, 
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
            spectra = pd.DataFrame( np.random.normal(self.obs_data_df['F_lambda'] , self.obs_data_df['F_lambda_error'] ) ,
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
                                                    feature_values_obs_err = self.obs_data_df['F_lambda_error'].values,
                                                     feature_names_obs  = self.obs_data_df['wl'].values, 
                                                     feature_names_synthetic = self.wl['wl'].values,
                                                      bd_literature_dic = self.bd_literature_dic,
                                                     # feature_values_obs_error = fluxcal['F_lambda_error'].values,
                                            )
            self.bd_object_generated = bd_object_generated







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
            plot_predicted_vs_observed(training_datasets = self.training_dataset_df, 
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
                                  features = self.obs_data_df['wl'], 
                                  feature_values = self.bd_object.Fnu_obs_absolute,
                                  error = np.array(self.bd_object.Fnu_obs_absolute_err),
                              training_datasets = self.training_dataset_df,  
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
                                        features = self.obs_data_df['wl'], 
                                        feature_values = self.bd_object.Fnu_obs_absolute,
                                        error = np.array(self.bd_object.Fnu_obs_absolute_err),
                                        training_datasets = self.training_dataset_df,  
                                        wl = self.wl,
                                        predicted_targets_dic = targets,
                                        bd_object_class = self.bd_object,
                                        print_results_ = True,
                                        )
                
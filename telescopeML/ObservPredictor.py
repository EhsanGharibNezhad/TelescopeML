# Import functions/Classes from other modules ====================

from io_funs import LoadSave

from StatVisAnalyzer import plot_predicted_vs_observed, boxplot_hist, plot_spectra_errorbar, plot_predicted_vs_spectra_errorbar
from StatVisAnalyzer import interpolate_df, print_results_fun
from StatVisAnalyzer import replace_zeros_with_mean, calculate_confidence_intervals_std_df, plot_predictedRandomSpectra_vs_ObservedSpectra_errorbar


# Import Python libraries ========================================

# Standard Data Manipulation / Statistical Libraries ******
import pandas as pd
import numpy as np
import pickle as pk
from scipy import stats

import os
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.io import output_notebook

# Data Visulaization Libraries ****************************
import matplotlib.pyplot as plt
import seaborn as sns
from uncertainties import ufloat


# Data science / Machine learning Libraries ***************
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from bokeh.palettes import Category20, colorblind
from tensorflow.keras.models import Sequential, model_from_json




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
                                    bd_literature_dic=None):
        """
        Process the observational dataset and set various attributes of the object.

        Args:
            feature_values_obs (array-like): Observed feature values.
            feature_values_obs_err (array-like): Errors corresponding to the observed feature values.
            feature_names_obs (array-like): Names of the observed features.
            feature_names_synthetic (array-like): Names of the synthetic features.
            bd_literature_dic (dict): Dictionary containing literature information. Defaults to None.
        """

        self.feature_values_obs = feature_values_obs
        self.feature_values_obs_err = feature_values_obs_err
        self.feature_names_obs = feature_names_obs
        self.feature_names_model = np.sort(feature_names_synthetic)
        self.bd_literature_dic = bd_literature_dic

        self.Fnu_obs = self.Flam2Fnu()[0]  # Convert Flam_obs to Fnu_obs
        self.Fnu_obs_err = self.Flam2Fnu()[1]  # Convert Flam_obs to Fnu_obs

        if bd_literature_dic['bd_distance_pc'] is not None and bd_literature_dic['bd_radius_Rjup'] is not None:
            # Calculate the absolute Fnu_obs values
            self.Fnu_obs_absolute = self.Fnu_obs * (bd_literature_dic['bd_distance_pc'] * ((u.pc).to(u.jupiterRad)) / (bd_literature_dic['bd_radius_Rjup'])) ** 2
            self.Fnu_obs_absolute_err = self.Fnu_obs_err * (bd_literature_dic['bd_distance_pc'] * ((u.pc).to(u.jupiterRad)) / (bd_literature_dic['bd_radius_Rjup'])) ** 2

        
    def Flam2Fnu(self):
        """
        Convert F_lambda to F_nu along with error propagation.

        Returns:
            fnu_values (array-like): Array of flux density values in F_nu.
            fnu_errors (array-like): Array of error bars for the flux density values in F_nu.
        """
        flam_values = self.feature_values_obs
        flam_errors = self.feature_values_obs_err
        wavelengths = self.feature_names_obs

        flam_dataset = NDDataArray(flam_values, uncertainty=StdDevUncertainty(flam_errors), unit=(u.erg / u.s / u.Angstrom / u.cm**2))

        Fnu_all = flam_dataset.convert_unit_to(unit=(u.erg / u.s / u.cm**2 / u.Hz), equivalencies=u.spectral_density(wavelengths * u.micron))
        fnu_values = Fnu_all.data
        fnu_errors = Fnu_all.uncertainty.array

        return fnu_values, fnu_errors

        
    def flux_interpolated(self, 
                          __print_results__ = False, 
                          __plot_spectra_errorbar__  = False, 
                          use_spectres = True):
        """
        Perform flux interpolation using either SpectRes or pchip interpolation.

        Args:
            __print_results__ (bool): Whether to print the results. Defaults to False.
            __plot_results__ (bool): Whether to plot the results. Defaults to True.
            use_spectres (bool): Whether to use SpectRes for interpolation. Defaults to True.
        """
        if use_spectres:
            self.Fnu_obs_absolute_intd = spectres.spectres(self.feature_names_model,
                                                           np.float64(self.feature_names_obs),
                                                           np.float64(self.Fnu_obs_absolute))
        else:
            flux_intd = pchip(self.feature_names_obs, self.Fnu_obs_absolute)
            self.Fnu_obs_absolute_intd = flux_intd(self.feature_names_model)

        self.df_Fnu_obs_absolute_intd = pd.DataFrame(list(self.Fnu_obs_absolute_intd),
                                           index=[str(x) for x in np.round(self.feature_names_model, 3)]).T

        if __print_results__:
            print('---    Object Flux     ----')
            print(self.df_Fnu_obs_absolute_intd)
            print('---    Object Flux Interpolate     ----')
            print(pd.DataFrame(self.Fnu_obs_absolute_intd))

        if __plot_spectra_errorbar__:
            plot_spectra_errorbar(
                        object_name = self.object_name,
                        x_obs = self.obs_data_df['wl'], 
                        y_obs = self.obs_data_df['F_lambda'],
                        y_obs_err = self.obs_data_df['F_lambda_error'],
                        y_label = 'Flux (Flam)',
                                )   
            
            plot_spectra_errorbar(
                        object_name = self.object_name,
                        x_obs = self.obs_data_df['wl'], 
                        y_obs = self.Fnu_obs,
                        y_obs_err = self.Fnu_obs_err,
                        y_label = 'Flux (F𝜈)',
                                )   
        
            plot_spectra_errorbar(
                        object_name = self.object_name,
                        x_obs = self.obs_data_df['wl'], 
                        y_obs = self.Fnu_obs_absolute,
                        y_obs_err = self.Fnu_obs_absolute_err,
                        y_label='Flux_abs (F_abs)',
                                )   
        
        

    def load_observational_spectra(self, 
                                   __plot_observational_spectra_errorbar__= False, 
                                   _replace_zeros_with_mean_=True,
                                   __print_results__ = False,
                                  ):
        """
        Load the observational spectra, process the dataset, and optionally plot the observational spectra with error bars.

        Args:
            __plot_observational_spectra_errorbar__ (bool): Whether to plot the observational spectra with error bars. Defaults to True.
            _replace_zeros_with_mean_ (bool): Whether to replace zeros with the mean of their non-zero neighbors. Defaults to True.
        """
        # Load the observational spectra
        obs_data_df = pd.read_csv(f'../../datasets/observational_spectra/{self.object_name}_fluxcal.dat',
                                 delim_whitespace=True, comment='#', names=('wl', 'F_lambda', 'F_lambda_error'),
                                 usecols=(0, 1, 2))

        # Process the dataset - remove Nan and negative values
        obs_data_df['F_lambda'] = obs_data_df['F_lambda'].mask(obs_data_df['F_lambda'].lt(0), 0)
        obs_data_df['F_lambda'].replace(0, np.nan, inplace=True)
        obs_data_df['F_lambda'].interpolate(inplace=True)

        if _replace_zeros_with_mean_:
            obs_data_df['F_lambda_error'] = replace_zeros_with_mean(obs_data_df['F_lambda_error'])
            # obs_data_df['F_lambda'] = replace_zeros_with_mean(obs_data_df['F_lambda']) # gives wrong numbers

        self.obs_data_df = obs_data_df
        
        if __print_results__:
            print(self.obs_data_df.head(5))

        if __plot_observational_spectra_errorbar__:
            plot_spectra_errorbar(
                        object_name = self.object_name,
                        x_obs = self.obs_data_df['wl'], 
                        y_obs = self.obs_data_df['F_lambda'],
                        y_obs_err = self.obs_data_df['F_lambda_error'],
                        y_label='Flux (Flam)',
                                )   
            # plot_spectra_errorbar(object_name=self.object_name,
            #                       features=self.obs_data_df['wl'],
            #                       feature_values=self.obs_data_df['F_lambda'],
            #                       error=self.obs_data_df['F_lambda_error'])

    
    
    def Process_Observational_Dataset(self, 
                                      __print_results__=False, 
                                      __plot_predicted_vs_observed__=False):
        """
        Process the observational dataset, extract ML features, perform predictions, and optionally print the results and plot the predicted versus observed spectra.

        Args:
            __print_results__ (bool): Whether to print the results. Defaults to True.
            __plot_predicted_vs_observed__ (bool): Whether to plot the predicted versus observed spectra. Defaults to True.
        """

        # Instantiate ProcessObservationalDataset class
        bd_object = self.ProcessObservationalDataset(
                    feature_values_obs=self.obs_data_df['F_lambda'].values,
                    feature_values_obs_err=self.obs_data_df['F_lambda_error'].values,
                    feature_names_obs=self.obs_data_df['wl'].values,
                    feature_names_synthetic=self.wl['wl'].values,
                    bd_literature_dic=self.bd_literature_dic,
        )

        # Extract the original ML features from the observational spectrum
        self.flux_interpolated(__print_results__ = False, 
                               __plot_spectra_errorbar__  = False, 
                               use_spectres = False)

        if __print_results__:
            print('------------ df_Fnu_obs_absolute_intd ------------')
            print(self.df_Fnu_obs_absolute_intd)

        # Extract the engineered ML features from the observational spectrum
        df_Fnu_obs_absolute_intd_min = self.df_Fnu_obs_absolute_intd.min(axis=1)
        df_Fnu_obs_absolute_intd_max = self.df_Fnu_obs_absolute_intd.max(axis=1)

        self.df_MinMax_obs = pd.DataFrame((df_Fnu_obs_absolute_intd_min, df_Fnu_obs_absolute_intd_max)).T

        if __print_results__:
            print('------------ df_MinMax Single Observational Spectrum ------------')
            print(self.df_MinMax_obs)

        XminXmax_Stand = self.train_cnn_regression_class.standardize_X_ColumnWise.transform(self.df_MinMax_obs.values)

        bd_mean = self.df_Fnu_obs_absolute_intd.mean(axis=1)[0]
        bd_std = self.df_Fnu_obs_absolute_intd.std(axis=1)[0]

        X_Scaled = (self.df_Fnu_obs_absolute_intd.values[0] - bd_mean) / bd_std

        y_pred_train = np.array(self.train_cnn_regression_class.trained_model.predict([X_Scaled[::-1].reshape(1, 104), XminXmax_Stand], verbose=0))[:, :, 0].T
        y_pred_train_ = self.train_cnn_regression_class.standardize_y_ColumnWise.inverse_transform(y_pred_train)
        y_pred_train_[:, 3] = 10 ** y_pred_train_[:, 3]
        y_pred = y_pred_train_
        self.y_pred = y_pred
        self.targets_single_spectrum_dic = dict(zip(['gravity', 'c_o_ratio', 'metallicity', 'temperature'], 
                                                    [np.round(self.y_pred[0][i],2) for i in range(4)]))
        # self.bd_object = bd_object

        if __print_results__:
            print_results_fun(targets=dict(zip(['bd_radius_Rjup', 'BD_mean', 'BD_std'], 
                                               [self.bd_literature_dic['bd_radius_Rjup'], bd_mean, bd_std])), 
                                      print_title='Radius, BD_mean, BD_std:')
            
            # print('Radius, BD_mean, BD_std:',self.bd_literature_dic['bd_radius_Rjup'], bd_mean, bd_std)
            
            print_results_fun(targets=self.targets_single_spectrum_dic, 
                              print_title='Predicted Targets from the Signle Observational Spectrum:')

        if __plot_predicted_vs_observed__:
            plot_predicted_vs_observed(
                training_datasets=self.training_dataset_df,
                wl=self.wl,
                predicted_targets_dic=self.targets_single_spectrum_dic,
                object_name=self.object_name,
                df_Fnu_obs_absolute_intd=self.df_Fnu_obs_absolute_intd,
            )



            
    def predict_from_random_spectra(
            self,
            random_spectra_num=10,
            __print_results__= False,
            __plot_randomly_generated_spectra__= False,
            __plot_histogram__= False,
            __plot_boxplot_hist__= False,
            __plot_predicted_vs_observed__= False,
            __plot_predicted_vs_spectra_errorbar__= False,
            __plot_predictedRandomSpectra_vs_ObservedSpectra_errorbar__= False,
            __calculate_confidence_intervals_std_df__=False,
        ):
        """
        - Generate random spectra based on the observational data, 
        - Perform predictions on the generated spectra
        - Provide various plotting and analysis options.

        Args:
            random_spectra_num (int): Number of random spectra to generate. Defaults to 10.
            __print_results__ (bool): Whether to print the results. Defaults to True.
            __plot_randomly_generated_spectra__ (bool): Whether to plot the randomly generated spectra. Defaults to True.
            __plot_histogram__ (bool): Whether to plot the histogram of predicted parameters. Defaults to True.
            __plot_boxplot_hist__ (bool): Whether to plot the boxplot and histogram of predicted parameters. Defaults to True.
            __plot_predicted_vs_observed__ (bool): Whether to plot the predicted versus observed spectra. Defaults to True.
            __plot_predicted_vs_spectra_errorbar__ (bool): Whether to plot the predicted versus observed spectra with error bars. Defaults to True.
            __plot_predictedRandomSpectra_vs_ObservedSpectra_errorbar__ (bool): Whether to plot the predicted random spectra versus observed spectra with error bars. Defaults to True.
            __calculate_confidence_intervals_std_df__ (bool): Whether to calculate and plot confidence intervals and standard deviations. Defaults to True.
        """

        color = viridis(250).__iter__()

        spectra_list_obs = []
        spectra_list_pre = []
        param_list = []

        # Generate random spectra
        for i in range(random_spectra_num):
            spectra = pd.DataFrame(
                np.random.normal(self.obs_data_df['F_lambda'], self.obs_data_df['F_lambda_error']),
                columns=['F_lambda']
            )

            # Process the dataset: fix negatives and nans
            spectra['F_lambda'] = spectra['F_lambda'].mask(spectra['F_lambda'].lt(0), 0)
            spectra['F_lambda'].replace(0, np.nan, inplace=True)
            spectra['F_lambda'].interpolate(inplace=True)

            # Process the randomly generated Observational spectra
            bd_object_generated = self.ProcessObservationalDataset(
                                        feature_values_obs = spectra['F_lambda'].values,
                                        feature_values_obs_err = self.obs_data_df['F_lambda_error'].values,
                                        feature_names_obs = self.obs_data_df['wl'].values,
                                        feature_names_synthetic = self.wl['wl'].values,
                                        bd_literature_dic = self.bd_literature_dic,
                                    )

            # Extract the original ML features from the observational spectrum
            self.flux_interpolated(__print_results__ = False, 
                                   __plot_spectra_errorbar__ = False, 
                                   use_spectres = True
                                  )

            # Extract the engineered ML features from the observational spectrum
            df_Fnu_obs_absolute_intd_min = self.df_Fnu_obs_absolute_intd.min(axis=1)
            df_Fnu_obs_absolute_intd_max = self.df_Fnu_obs_absolute_intd.max(axis=1)

            df_MinMax_obs = pd.DataFrame(
                (df_Fnu_obs_absolute_intd_min, df_Fnu_obs_absolute_intd_max)
                ).T

            XminXmax_Stand = self.train_cnn_regression_class.standardize_X_ColumnWise.transform(df_MinMax_obs.values)

            bd_mean = self.df_Fnu_obs_absolute_intd.mean(axis=1)[0]
            bd_std = self.df_Fnu_obs_absolute_intd.std(axis=1)[0]

            X_Scaled = (self.df_Fnu_obs_absolute_intd.values[0] - bd_mean) / bd_std

            y_pred_train = np.array(
                self.train_cnn_regression_class.trained_model.predict(
                    [X_Scaled[::-1].reshape(1, 104), XminXmax_Stand], verbose=0)
                )[:, :, 0].T
            
            y_pred_train_ = self.train_cnn_regression_class.standardize_y_ColumnWise.inverse_transform(y_pred_train)
            y_pred_train_[:, 3] = 10 ** y_pred_train_[:, 3]
            y_pred_random = y_pred_train_

            targets_dic_random = dict(zip(['gravity', 'c_o_ratio', 'metallicity', 'temperature'], 
                                          [y_pred_random[0][i] for i in range(4)])
                                        )

            spectra_list_obs.append(self.df_Fnu_obs_absolute_intd.values)
            param_list.append(y_pred_random[0])


            filtered_df4 = interpolate_df(dataset = self.training_dataset_df, 
                                         predicted_targets_dic = targets_dic_random, #self.targets_single_spectrum_dic,# 
                                         print_results_ = False)
            # self.filtered_df4 = filtered_df4
            # print(filtered_df4.iloc[0,0:-5].values)
            
            # if __print_results__: FINDME

        

            spectra_list_pre.append( filtered_df4.iloc[:,0:-5].values.flatten() )
            


        # print('*'*10+'  Filtered and Interpolated training data based on the ML predicted parameters  '+'*'*10)
        # print(spectra_list_pre)
            
            
        self.spectra_list_obs = spectra_list_obs
        self.spectra_list_pre = spectra_list_pre
        self.param_list = param_list

        self.df_random_pred = pd.DataFrame(self.param_list, columns=['logg', 'c_o', 'met', 'T'])

        self.dic_random_pred_mean = dict(
            zip(['gravity', 'c_o_ratio', 'metallicity', 'temperature'], list(self.df_random_pred.agg(np.mean)))
        )
        
        # self.df_spectra_list_obs = pd.DataFrame(data=np.array(self.spectra_list_obs).reshape(-1, 104), columns=self.wl.wl)
        self.df_spectra_list_pre = pd.DataFrame(data=self.spectra_list_pre, columns=self.wl.wl[::-1])
        # print(self.spectra_list_pre)
        # print(self.df_spectra_list_obs)
        
        if __print_results__:
            print_results_fun(targets=self.dic_random_pred_mean, 
                              print_title='Predicted Targets from Randomly Generated Spectra:')

        if __print_results__:
            print(self.df_random_pred.describe())

        if __plot_randomly_generated_spectra__:
            p = figure(
                title=self.object_name + ": Randomly generated spectra within 1σ",
                x_axis_label='Features (Wavelength [𝜇m])',
                y_axis_label='Flux (Fν)',
                width = 800,
                height = 300,
                y_axis_type="log",
                background_fill_color="#fafafa",
            )

            if random_spectra_num > 250:
                random_spectra_num_index = 200
            else:
                random_spectra_num_index = random_spectra_num

            for i in range(0, random_spectra_num_index):
                p.line(
                    self.wl.wl.values[::-1],
                    self.spectra_list_obs[i][0],
                    line_width=1,
                    line_alpha=0.6,
                    line_color=next(color),
                )

            # Increase size of x and y ticks
            p.title.text_font_size = '12pt'
            p.xaxis.major_label_text_font_size = '12pt'
            p.xaxis.axis_label_text_font_size = '12pt'
            p.yaxis.major_label_text_font_size = '12pt'
            p.yaxis.axis_label_text_font_size = '12pt'

            show(p)

        if __plot_histogram__:
            plt.figure()
            self.df_random_pred.hist()
            plt.show()

        if __plot_boxplot_hist__:
            boxplot_hist(self.df_random_pred['logg'], x_label=r'$\log g$', xy_loc=[0.05, 0.98])
            boxplot_hist(self.df_random_pred['T'], x_label=r'$T_{eff}$', xy_loc=[0.05, 0.98])
            boxplot_hist(self.df_random_pred['c_o'], x_label=r'C/O', xy_loc=[0.05, 0.98])
            boxplot_hist(self.df_random_pred['met'], x_label=r'[M/H]', xy_loc=[0.05, 0.98])

        if __plot_predicted_vs_observed__:
            
            plot_predicted_vs_observed(
                training_datasets=self.training_dataset_df,
                wl=self.wl,
                predicted_targets_dic = self.dic_random_pred_mean,
                object_name=self.object_name,
                df_Fnu_obs_absolute_intd=self.df_Fnu_obs_absolute_intd,
                __print_results__= False,
            )

        if __plot_predicted_vs_spectra_errorbar__:
            
            plot_predicted_vs_spectra_errorbar(
                        object_name = self.object_name,
                        x_obs = self.obs_data_df['wl'],
                        y_obs = self.Fnu_obs_absolute,
                        y_obs_error = np.array(self.Fnu_obs_absolute_err),
                        training_dataset = self.training_dataset_df,
                        x_pred = self.wl,
                        predicted_targets_dic = self.dic_random_pred_mean,
                        __print_results__ = False,
            )

            
        if __calculate_confidence_intervals_std_df__:
            
            self.confidence_intervals_std_df = calculate_confidence_intervals_std_df(
                dataset_df = self.df_spectra_list_pre,
                __print_results__ = False,
                __plot_calculate_confidence_intervals_std_df__ = False,
            )

            
        if __plot_predictedRandomSpectra_vs_ObservedSpectra_errorbar__:
            
            plot_predictedRandomSpectra_vs_ObservedSpectra_errorbar(
                stat_df = self.confidence_intervals_std_df,
                confidence_level = 0.95,
                object_name = self.object_name,
                x_obs = self.obs_data_df['wl'],
                y_obs = self.Fnu_obs_absolute,
                y_obs_err = np.array(self.Fnu_obs_absolute_err),
                training_datasets = self.training_dataset_df,
                x_pred = self.wl,
                predicted_targets_dic = self.dic_random_pred_mean, #self.dic_random_pred_mean,
                __print_results__ = False,
            )

            

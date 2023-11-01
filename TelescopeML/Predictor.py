# ======= Import functions/Classes from other modules ====================

from .IO_utils import *

from .StatVisAnalyzer import boxplot_hist, plot_spectra_errorbar, \
    plot_pred_vs_obs_errorbar
from .StatVisAnalyzer import interpolate_df, print_results_fun
from .StatVisAnalyzer import replace_zeros_with_mean, calculate_confidence_intervals_std_df, \
    plot_pred_vs_obs_errorbar_stat

# ======= Import Python libraries ========================================

# Standard Data Manipulation / Statistical Libraries ******
# import os
import pandas as pd
import numpy as np
# import pickle as pk
# from scipy import stats
# from uncertainties import ufloat

import astropy.units as u
from scipy.interpolate import pchip
# from tensorflow.keras.models import Sequential, model_from_json
import spectres

# Data Visulaization Libraries ****************************
import matplotlib.pyplot as plt
# import seaborn as sns

from bokeh.io import output_notebook

output_notebook()
from bokeh.plotting import show, figure

TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
]

# Data science / Machine learning Libraries ***************

# from astropy.constants import c
from astropy.nddata import StdDevUncertainty, NDDataArray
from bokeh.palettes import viridis


__reference_data__ = os.getenv("TelescopeML_reference_data")
print(__reference_data__)

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


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# Data Visualizing libararies
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


# ===============================================================================
# ========================                           ============================
# ========================    observational_spectra  ============================
# ========================                           ============================
# ===============================================================================


class ObserveParameterPredictor:
    """
    Load, process, and visualize observational spectra.

    Parameters
    ----------
    F_lambda_obs : array
        Fluxes for each feature (wavelength) from observational data.
    F_lambda_obs_err : array
        Observed spectra error bars.
    wl_obs : array
        Name of features (wavelength) from observational data, e.g., 0.9, 1.0, 1.1 micron.
    wl_synthetic : array
        Name of features (wavelengths) from synthetic data.

    Example
    -------
        >>> HD3651B_BD_literature_info = {
        >>>   'bd_name':'HD3651B',
        >>>   'bd_Teff':818,
        >>>   'bd_logg':3.94,
        >>>   'bd_met': -0.22,
        >>>   'bd_distance_pc' : 11.134,
        >>>   'bd_radius_Rjup' : 0.81,
        >>>   'bd_radius_Rjup_tuned': .81}
    """

    def __init__(self,
                 object_name,
                 training_dataset_df,
                 wl_synthetic,
                 BuildRegressorCNN_class,
                 bd_literature_dic,
                 ):

        self.object_name = object_name
        self.training_dataset_df = training_dataset_df
        self.wl_synthetic = wl_synthetic
        self.BuildRegressorCNN_class = BuildRegressorCNN_class
        self.bd_literature_dic = bd_literature_dic


    def load_observational_spectra(self,
                                   obs_data_df = None,
                                   __plot_observational_spectra_errorbar__=False,
                                   __replace_zeros_negatives_with_mean__=True,
                                   __print_results__=False,
                                   ):
        """
        Load the observational spectra, process the dataset, and optionally plot the observational spectra with error bars.

        Parameters
        -----------
        __plot_observational_spectra_errorbar__ : bool
            True or False.
        __replace_zeros_negatives_with_mean__ : bool
            True or False.
        __print_results__ : bool
            True or False.
        """

        # Load the observational spectra
        if obs_data_df is None:
            obs_data_df = pd.read_csv(__reference_data__+f'/observational_datasets/{self.object_name}_fluxcal.dat',
                                      delim_whitespace=True, comment='#', names=('wl', 'F_lambda', 'F_lambda_error'),
                                      usecols=(0, 1, 2))

        # # Process the dataset - remove Nan and negative values
        # obs_data_df['F_lambda'] = obs_data_df['F_lambda'].mask(obs_data_df['F_lambda'].lt(0), 0)
        # obs_data_df['F_lambda'].replace(0, np.nan, inplace=True)
        # obs_data_df['F_lambda'].interpolate(inplace=True)

        if __replace_zeros_negatives_with_mean__:
            obs_data_df['F_lambda_obs'] = obs_data_df['F_lambda'].mask(obs_data_df['F_lambda'].lt(0), 0)
            obs_data_df['F_lambda_obs'].replace(0, np.nan, inplace=True)
            obs_data_df['F_lambda_obs'].interpolate(method='linear', limit_direction='both', inplace=True)

            obs_data_df['F_lambda_obs_err'] = obs_data_df['F_lambda_error'].mask(obs_data_df['F_lambda_error'].lt(0), 0)
            obs_data_df['F_lambda_obs_err'].replace(0, np.nan, inplace=True)
            obs_data_df['F_lambda_obs_err'].interpolate(method='linear', limit_direction='both', inplace=True)

        self.obs_data_df = obs_data_df

        if __print_results__:
            print('------- Observational DataFrame Example ---------')
            print(self.obs_data_df.head(5))

        if __plot_observational_spectra_errorbar__:
            plot_spectra_errorbar(
                object_name=self.object_name,
                x_obs=self.obs_data_df['wl'],
                y_obs=self.obs_data_df['F_lambda_obs'],
                y_obs_err=self.obs_data_df['F_lambda_obs_err'],
                y_label='Flux (F𝛌) [erg/s/A/cm2]',
            )


    def ProcessObservationalDataset(self,
                                    F_lambda_obs,
                                    F_lambda_obs_err,
                                    wl_obs,
                                    # wl_synthetic,
                                    bd_literature_dic=None):
        """
        Process the observational dataset and set various attributes of the object.

        Parameters
        -----------
        F_lambda_obs : array
            Observed feature values.
        F_lambda_obs_err : array
            Errors corresponding to the observed feature values.
        wl_obs : array
            Names of the observed features.
        wl_synthetic : array
            Names of the synthetic features.
        bd_literature_dic : dict, optional
            Dictionary containing literature information. Defaults to None.

        Return
        -------
        Fnu_obs , Fnu_obs_err, Fnu_obs_absolute, Fnu_obs_absolute_err

        Example
        -------

        >>> HD3651B_BD_literature_info = {'bd_name':'HD3651B',
        >>>   'bd_Teff':818,
        >>>   'bd_logg':3.94,
        >>>   'bd_met': -0.22,
        >>>   'bd_distance_pc' : 11.134,
        >>>   'bd_radius_Rjup' : 0.81,
        >>>   'bd_radius_Rjup_tuned': .81}

        """

        self.F_lambda_obs = F_lambda_obs
        self.F_lambda_obs_err = F_lambda_obs_err
        self.wl_obs = wl_obs
        # self.wl_synthetic = wl_synthetic
        self.bd_literature_dic = bd_literature_dic

        Fnu_obs , Fnu_obs_err = self.Flam_to_Fnu(
                 Flam_values = self.F_lambda_obs,
                 Flam_errors = self.F_lambda_obs_err,
                 wavelengths = wl_obs,)  # Convert F_lambda_obs to Fnu_obs

        Fnu_obs_absolute, Fnu_obs_absolute_err = self.Fnu_to_Fnu_abs(
                 Fnu_values = Fnu_obs,
                 Fnu_errors = Fnu_obs_err,
                 bd_literature_dic = bd_literature_dic)

        # self.obs_data_df['Fnu_obs'] = self.Fnu_obs
        # self.obs_data_df['Fnu_obs_err'] = self.Fnu_obs_err
        # self.obs_data_df['Fnu_obs_absolute'] = self.Fnu_obs_absolute
        # self.obs_data_df['Fnu_obs_absolute_err'] = self.Fnu_obs_absolute_err

        return Fnu_obs , Fnu_obs_err, Fnu_obs_absolute, Fnu_obs_absolute_err


    def Fnu_to_Fnu_abs(self,
                       Fnu_values,
                       Fnu_errors,
                       bd_literature_dic = None,
                       __plot__ = False
                       ):

        # Calculate the absolute Fnu_obs values
        bd_literature_dic = self.bd_literature_dic if bd_literature_dic is None else bd_literature_dic

        Fnu_values_abs = Fnu_values * (bd_literature_dic['bd_distance_pc'] * ((u.pc).to(u.jupiterRad)) / (
        bd_literature_dic['bd_radius_Rjup'])) ** 2

        Fnu_errors_abs = Fnu_errors * (
                    bd_literature_dic['bd_distance_pc'] * ((u.pc).to(u.jupiterRad)) / (
            bd_literature_dic['bd_radius_Rjup'])) ** 2

        if __plot__:
            plot_spectra_errorbar(
                object_name=self.object_name,
                x_obs=self.wl_obs,
                y_obs=Fnu_values_abs,
                y_obs_err=Fnu_errors_abs,
                y_label='Abs. Flux (F𝜈) [erg/s/cm2/Hz]',
            )

        return Fnu_values_abs, Fnu_errors_abs



    def Flam_to_Fnu(self,
                 Flam_values,
                 Flam_errors,
                 wavelengths):
        """
        Convert F_lambda to F_nu along with error propagation.
        Parameters
        ----------
        Flam_values
        Flam_errors
        wavelengths

        Returns
        -------
        fnu_values : array
            Array of flux density values in F_nu.
        fnu_errors : array
            Array of error bars for the flux density values in F_nu.
        """

        # flam_values = self.F_lambda_obs
        # flam_errors = self.F_lambda_obs_err
        # wavelengths = self.wl_obs

        flam_dataset = NDDataArray(Flam_values, uncertainty=StdDevUncertainty(Flam_errors),
                                   unit=(u.erg / u.s / u.Angstrom / u.cm ** 2))

        Fnu_all = flam_dataset.convert_unit_to(unit=(u.erg / u.s / u.cm ** 2 / u.Hz),
                                               equivalencies=u.spectral_density(wavelengths * u.micron))
        Fnu_values = Fnu_all.data
        Fnu_errors = Fnu_all.uncertainty.array

        return Fnu_values, Fnu_errors

    def flux_interpolated(self,
                          Fnu_obs_absolute,
                          interpolated_wl = None,
                          __print_results__=False,
                          __plot_spectra_errorbar__=False,
                          __use_spectres__=True):
        """
        Perform flux interpolation using either SpectRes or pchip interpolation.

        Parameters
        ----------
        __print_results__ : bool
            True or False.
        __plot_spectra_errorbar__ : bool
            True or False.
        __use_spectres__ : bool, optional
            Whether to use SpectRes for interpolation. Defaults to True.

        Return
        -------
        Fnu_obs_absolute_intd, Fnu_obs_absolute_intd_df
        """
        Fnu_obs_absolute = self.Fnu_obs_absolute if Fnu_obs_absolute is None else Fnu_obs_absolute
        interpolated_wl = np.sort(self.wl_synthetic.wl.values) if interpolated_wl is None else interpolated_wl

        if __use_spectres__:
            Fnu_obs_absolute_intd = spectres.spectres(interpolated_wl,
                                                           np.float64(self.wl_obs),
                                                           np.float64(Fnu_obs_absolute))
        else:
            flux_intd = pchip(self.wl_obs, self.Fnu_obs_absolute)
            Fnu_obs_absolute_intd = flux_intd(interpolated_wl)

        Fnu_obs_absolute_intd_df = pd.DataFrame(list(Fnu_obs_absolute_intd),
                                                     index=[str(x) for x in np.round(interpolated_wl, 3)]).T


        if __print_results__:
            print('---    Object Flux     ----')
            print(self.Fnu_obs_absolute_intd_df)
            print('---    Object Flux Interpolate     ----')
            print(pd.DataFrame(self.Fnu_obs_absolute_intd))

        if __plot_spectra_errorbar__:
            plot_spectra_errorbar(
                object_name=self.object_name,
                x_obs=self.obs_data_df['wl'],
                y_obs=self.obs_data_df['F_lambda_obs'],
                y_obs_err=self.obs_data_df['F_lambda_obs_err'],
                y_label='Flux (F𝛌)',
            )

            plot_spectra_errorbar(
                object_name=self.object_name,
                x_obs=self.obs_data_df['wl'],
                y_obs=self.Fnu_obs,
                y_obs_err=self.Fnu_obs_err,
                y_label='Flux (F𝜈)',
            )

            plot_spectra_errorbar(
                object_name=self.object_name,
                x_obs=self.obs_data_df['wl'],
                y_obs=self.Fnu_obs_absolute,
                y_obs_err=self.Fnu_obs_absolute_err,
                y_label='Flux_abs (F_abs)',
            )


        return Fnu_obs_absolute_intd, Fnu_obs_absolute_intd_df




    def Process_Observational_Dataset(self,
                                      __print_results__=False,
                                      F_lambda_obs = None,
                                      F_lambda_obs_err = None,
                                      # __plot_predicted_vs_observed__=False
                                      ):
        """
        Process the observational dataset, extract ML features, perform predictions, and optionally print the results and plot the predicted versus observed spectra.

        Parameters
        ----------
        __print_results__ : bool
            True or False.
        """


        F_lambda_obs = self.obs_data_df['F_lambda_obs'] if F_lambda_obs == None else F_lambda_obs
        F_lambda_obs_err = self.obs_data_df['F_lambda_obs_err'] if F_lambda_obs_err == None else F_lambda_obs_err

        Fnu_obs , Fnu_obs_err, Fnu_obs_absolute, Fnu_obs_absolute_err = self.ProcessObservationalDataset(
                    F_lambda_obs=F_lambda_obs.values,
                    F_lambda_obs_err=F_lambda_obs_err.values,
                    wl_obs=self.obs_data_df['wl'].values,
                    # wl_synthetic=self.wl_synthetic['wl'].values,
                    bd_literature_dic=self.bd_literature_dic,
                   )

        # self.Fnu_obs, Fnu_obs_err, self.Fnu_obs_absolute, self.Fnu_obs_absolute_err = Fnu_obs , Fnu_obs_err, Fnu_obs_absolute, Fnu_obs_absolute_err
        self.obs_data_df['Fnu_obs'] = Fnu_obs
        self.obs_data_df['Fnu_obs_err'] = Fnu_obs_err
        self.obs_data_df['Fnu_obs_absolute'] = Fnu_obs_absolute
        self.obs_data_df['Fnu_obs_absolute_err'] = Fnu_obs_absolute_err

        # Extract the original ML features from the observational spectrum
        self.Fnu_obs_absolute_intd, self.Fnu_obs_absolute_intd_df = \
                    self.flux_interpolated(Fnu_obs_absolute = Fnu_obs_absolute,
                                           interpolated_wl = None,
                                           __print_results__=False,
                                           __plot_spectra_errorbar__=False,
                                           __use_spectres__=True)

        if __print_results__:
            print('------------  Interpolated Observational Spectra: Absolute F𝜈 ------------')
            print(self.Fnu_obs_absolute_intd_df)

        # Extract the engineered ML features from the observational spectrum
        df_Fnu_obs_absolute_intd_min = self.Fnu_obs_absolute_intd_df.min(axis=1)
        df_Fnu_obs_absolute_intd_max = self.Fnu_obs_absolute_intd_df.max(axis=1)

        self.df_MinMax_obs = pd.DataFrame((df_Fnu_obs_absolute_intd_min, df_Fnu_obs_absolute_intd_max)).T

        if __print_results__:
            print('------------ df_MinMax Single Observational Spectrum ------------')
            print(self.df_MinMax_obs)

        # this is commited b/c it is for standardize_X_ColumnWise for the MinMax
        XminXmax_Stand = self.BuildRegressorCNN_class.standardize_X_ColumnWise.transform(self.df_MinMax_obs.values)

        # XminXmax_Stand = self.BuildRegressorCNN_class.normalize_X_ColumnWise.transform(self.df_MinMax_obs.values)

        bd_mean = self.Fnu_obs_absolute_intd_df.mean(axis=1)[0]
        bd_std = self.Fnu_obs_absolute_intd_df.std(axis=1)[0]

        X_Scaled = (self.Fnu_obs_absolute_intd_df.values[0] - bd_mean) / bd_std


        y_pred_train = np.array(
            self.BuildRegressorCNN_class.trained_model.predict([X_Scaled[::-1].reshape(1, 104), XminXmax_Stand],
                                                                  verbose=0))[:, :, 0].T
        y_pred_train_ = self.BuildRegressorCNN_class.standardize_y_ColumnWise.inverse_transform(y_pred_train)
        y_pred_train_[:, 3] = 10 ** y_pred_train_[:, 3]
        y_pred = y_pred_train_
        self.y_pred = y_pred
        self.targets_single_spectrum_dic = dict(zip(['gravity', 'c_o_ratio', 'metallicity', 'temperature'],
                                                    [np.round(self.y_pred[0][i], 2) for i in range(4)]))
        # self.bd_object = bd_object

        if __print_results__:
            print_results_fun(targets=dict(zip(['bd_radius_Rjup', 'BD_mean', 'BD_std'],
                                               [self.bd_literature_dic['bd_radius_Rjup'], bd_mean, bd_std])),
                              print_title='Radius, BD_mean, BD_std:')

            # print('Radius, BD_mean, BD_std:',self.bd_literature_dic['bd_radius_Rjup'], bd_mean, bd_std)

            print_results_fun(targets=self.targets_single_spectrum_dic,
                              print_title='Predicted Targets from the Signle Observational Spectrum:')

        # if __plot_predicted_vs_observed__:
        #     plot_predicted_vs_observed(
        #         training_datasets=self.training_dataset_df,
        #         wl=self.wl,
        #         predicted_targets_dic=self.targets_single_spectrum_dic,
        #         object_name=self.object_name,
        #         Fnu_obs_absolute_intd_df=self.Fnu_obs_absolute_intd_df,
        #     )

    def predict_from_random_spectra(
            self,
            random_spectra_num=10,
            __print_results__=False,
            __plot_randomly_generated_spectra__=False,
            __plot_histogram__=False,
            __plot_boxplot_hist__=False,
            # __plot_predicted_vs_observed__=False,
            __plot_pred_vs_obs_errorbar__=False,
            __plot_pred_vs_obs_errorbar_stat__=False,
            __calculate_confidence_intervals_std_df__=False,
            ):
        """

        Generate random spectra based on the observational data, predict the target features, and plot the spectra

        Parameters
        ----------
        random_spectra_num : int, optional
            Number of random spectra to generate. Defaults to 10.
        __print_results__ : bool
            True or False.
        __plot_randomly_generated_spectra__ : bool
            True or False.
        __plot_histogram__ : bool
            True or False.
        __plot_boxplot_hist__ : bool
            True or False.
        __plot_pred_vs_obs_errorbar__ : bool
            True or False.
        __plot_pred_vs_obs_errorbar_stat__ : bool
            True or False.
        __calculate_confidence_intervals_std_df__ : bool
            True or False.
        """

        color = viridis(250).__iter__()

        spectra_list_obs = []
        spectra_list_pre = []
        param_list = []

        # Generate random spectra
        for i in range(random_spectra_num):
            spectra = pd.DataFrame(
                np.random.normal(self.obs_data_df['F_lambda_obs'], self.obs_data_df['F_lambda_obs_err']),
                columns=['F_lambda_obs']
            )

            # Process the dataset: fix negatives and nans
            spectra['F_lambda_obs'] = spectra['F_lambda_obs'].mask(spectra['F_lambda_obs'].lt(0), 0)
            spectra['F_lambda_obs'].replace(0, np.nan, inplace=True)
            spectra['F_lambda_obs'].interpolate(inplace=True)

            # Process the randomly generated Observational spectra
            Fnu_obs , Fnu_obs_err, Fnu_obs_absolute, Fnu_obs_absolute_err = self.ProcessObservationalDataset(
                    F_lambda_obs=spectra['F_lambda_obs'].values,
                    F_lambda_obs_err=self.obs_data_df['F_lambda_obs_err'].values,
                    wl_obs=self.obs_data_df['wl'].values,
                    # wl_synthetic=self.wl_synthetic['wl'].values,
                    bd_literature_dic=self.bd_literature_dic,
                    )

            # self.obs_data_df['Fnu_obs'] = self.Fnu_obs
            # self.obs_data_df['Fnu_obs_err'] = self.Fnu_obs_err
            # self.obs_data_df['Fnu_obs_absolute'] = self.Fnu_obs_absolute
            # self.obs_data_df['Fnu_obs_absolute_err'] = self.Fnu_obs_absolute_err

            # Extract the original ML features from the observational spectrum
            Fnu_obs_absolute_intd, Fnu_obs_absolute_intd_df = \
                        self.flux_interpolated(Fnu_obs_absolute = Fnu_obs_absolute,
                                               interpolated_wl = None,
                                               __print_results__=False,
                                               __plot_spectra_errorbar__=False,
                                               __use_spectres__=True)

            # Extract the engineered ML features from the observational spectrum
            Fnu_obs_absolute_intd_df_min = Fnu_obs_absolute_intd_df.min(axis=1)
            Fnu_obs_absolute_intd_df_max = Fnu_obs_absolute_intd_df.max(axis=1)

            df_MinMax_obs = pd.DataFrame(
                (Fnu_obs_absolute_intd_df_min, Fnu_obs_absolute_intd_df_max)
            ).T
            # print('Bug check1 -- df_MinMax_obs:', df_MinMax_obs)
            XminXmax_Stand = self.BuildRegressorCNN_class.standardize_X_ColumnWise.transform(df_MinMax_obs.values)
            # XminXmax_Stand = self.BuildRegressorCNN_class.normalize_X_ColumnWise.transform(df_MinMax_obs.values)

            # print('Bug check2 -- XminXmax_Stand:', XminXmax_Stand)


            bd_mean = Fnu_obs_absolute_intd_df.mean(axis=1)[0]
            bd_std = Fnu_obs_absolute_intd_df.std(axis=1)[0]

            # print('Bug check3 -- bd_mean, bd_std:', bd_mean, bd_std)

            # X_Scaled = (Fnu_obs_absolute_intd_df.div((self.bd_literature_dic['bd_radius_Rjup'])**2).values[0] - bd_mean) / bd_std
            X_Scaled = (Fnu_obs_absolute_intd_df.values[0] - bd_mean) / bd_std
            # print('Bug check4 -- X_Scaled:', X_Scaled)

            y_pred_train = np.array(
                self.BuildRegressorCNN_class.trained_model.predict(
                    [X_Scaled[::-1].reshape(1, 104), XminXmax_Stand.reshape(1, 2)], verbose=0) #findme!
            )[:, :, 0].T

            y_pred_train_ = self.BuildRegressorCNN_class.standardize_y_ColumnWise.inverse_transform(y_pred_train)
            y_pred_train_[:, 3] = 10 ** y_pred_train_[:, 3]
            y_pred_random = y_pred_train_

            targets_dic_random = dict(zip(['gravity', 'c_o_ratio', 'metallicity', 'temperature'],
                                          [y_pred_random[0][i] for i in range(4)])
                                      )


            spectra_list_obs.append(Fnu_obs_absolute_intd_df.values)
            param_list.append(y_pred_random[0])

            filtered_df4 = interpolate_df(dataset=self.training_dataset_df,
                                          predicted_targets_dic=targets_dic_random,
                                          # self.targets_single_spectrum_dic,#
                                          print_results_=False)
            # self.filtered_df4 = filtered_df4
            # print(filtered_df4.iloc[0,0:-5].values)

            # if __print_results__: FINDME

            spectra_list_pre.append(filtered_df4.iloc[:, 0:-5].div((self.bd_literature_dic['bd_radius_Rjup'])**2).values.flatten())
            # spectra_list_pre.append(filtered_df4.iloc[:, 0:-5].values.flatten())
            # print('Bug check5 -- spectra_list_pre:', spectra_list_pre)

        # print('*'*10+'  Filtered and Interpolated training data based on the ML predicted parameters  '+'*'*10)
        # print(spectra_list_pre)

        self.spectra_list_obs = spectra_list_obs
        self.spectra_list_pre = spectra_list_pre
        self.param_list = param_list

        self.df_random_pred = pd.DataFrame(self.param_list, columns=['logg', 'c_o', 'met', 'T'])

        self.dic_random_pred_mean = dict(
            zip(['gravity', 'c_o_ratio', 'metallicity', 'temperature'],
                list(self.df_random_pred.agg(np.mean)))
        )

        # self.df_spectra_list_obs = pd.DataFrame(data=np.array(self.spectra_list_obs).reshape(-1, 104), columns=self.wl_synthetic.wl_synthetic)
        self.df_spectra_list_pre = pd.DataFrame(data=self.spectra_list_pre, columns=self.wl_synthetic.wl[::-1])
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
                y_axis_label='Flux (F𝜈) [erg/s/cm2/Hz]',
                width=800,
                height=300,
                y_axis_type="log",
                background_fill_color="#fafafa",
            )

            if random_spectra_num > 250:
                random_spectra_num_index = 200
            else:
                random_spectra_num_index = random_spectra_num

            for i in range(0, random_spectra_num_index):
                p.line(
                    self.wl_synthetic.wl.values[::-1],
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

        # if __plot_predicted_vs_observed__:
        #     plot_predicted_vs_observed(
        #         training_datasets=self.training_dataset_df,
        #         wl=self.wl_synthetic,
        #         predicted_targets_dic=self.dic_random_pred_mean,
        #         object_name=self.object_name,
        #         Fnu_obs_absolute_intd_df=self.Fnu_obs_absolute_intd_df,
        #         __print_results__=False,
        #     )

        if __plot_pred_vs_obs_errorbar__:
            plot_pred_vs_obs_errorbar(
                object_name=self.object_name,
                x_obs=self.obs_data_df['wl'],
                y_obs=Fnu_obs_absolute,
                y_obs_error=Fnu_obs_absolute_err,
                training_dataset=self.training_dataset_df,
                x_pred=self.wl_synthetic,
                predicted_targets_dic=self.dic_random_pred_mean,
                __print_results__=False,
            )

        if __calculate_confidence_intervals_std_df__:
            self.confidence_intervals_std_df = calculate_confidence_intervals_std_df(
                dataset_df=self.df_spectra_list_pre,
                __print_results__=False,
                __plot_calculate_confidence_intervals_std_df__=False,
            )

        if __plot_pred_vs_obs_errorbar_stat__:
            plot_pred_vs_obs_errorbar_stat(
                stat_df=self.confidence_intervals_std_df,
                confidence_level=0.95,
                object_name=self.object_name,
                x_obs=self.obs_data_df['wl'],
                y_obs=self.obs_data_df['Fnu_obs_absolute'], #self.obs_data_df['Fnu_obs_absolute'],# self.Fnu_obs_absolute,
                y_obs_err=self.obs_data_df['Fnu_obs_absolute_err'], #self.obs_data_df['Fnu_obs_absolute_err'],#self.Fnu_obs_absolute_err,
                training_datasets=self.training_dataset_df,
                x_pred=self.wl_synthetic,
                predicted_targets_dic=self.dic_random_pred_mean,  # self.dic_random_pred_mean,
                radius = self.bd_literature_dic['bd_radius_Rjup'],
                __print_results__=False,
            )

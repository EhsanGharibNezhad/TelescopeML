# Import functions/Classes from other modules ====================

# from io_funs import LoadSave

# Import libraries ========================================

# ******* Standard Data Manipulation / Statistical Libraries *****
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
# import pickle as pk

from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import chi2
import os
from matplotlib.ticker import AutoMinorLocator

import pprint

# ******* Data Visulaization Libraries ****************************
import seaborn as sns
import matplotlib.pyplot as plt


from bokeh.plotting import output_notebook

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def print_results_fun(targets, print_title=None):
    """
    Print the outputs in a pretty format using the pprint library.

    Parameters
    ----------
    targets : any
        The data to be printed.
    print_title : str
        An optional title to display before the printed data.
    """

    print('*' * 30 + '\n')

    if print_title is not None:
        print(print_title+ '\n')

    # Use pprint to print the data in a well-formatted and indented manner.
    pprint.pprint(targets, indent=4, width=30)

    print('*' * 30 + '\n')




def filter_dataset_range(dataset, filter_params):
    """
    filer the dataframe
    """
    filtered_df = dataset.copy()

    for param, bounds in filter_params.items():
        lower_bound, upper_bound = bounds
        filtered_df = filtered_df[(filtered_df[param] >= lower_bound) & (filtered_df[param] <= upper_bound)]

    return filtered_df

def find_nearest_top_bottom(value, lst):
    """
    Find the nearest value in the list of data
    """
    lst.sort()
    nearest_top = None
    nearest_bottom = None

    for num in lst:
        if num >= value:
            nearest_top = num
            break

    if nearest_top is None:
        nearest_top = lst[-1]

    for num in reversed(lst):
        if num <= value:
            nearest_bottom = num
            break

    if nearest_bottom is None:
        nearest_bottom = lst[0]

    return nearest_bottom, nearest_top


def interpolate_df(dataset,
                   predicted_targets_dic,
                   print_results_=False):
    """
    Interpolate the training set.

    Parameters
    ----------
    dataset : array
        The training dataset to be interpolated.
    predicted_targets_dic : dict
        Target features to be interpolated.
    """

    my_list_g = list(dataset['gravity'].sort_values().unique())
    my_list_met = list(dataset['metallicity'].sort_values().unique())
    my_list_c_o = list(dataset['c_o_ratio'].sort_values().unique())
    my_list_T = list(dataset['temperature'].sort_values().unique())

    g0, g1 = find_nearest_top_bottom(predicted_targets_dic['gravity'], my_list_g)
    co0, co1 = find_nearest_top_bottom(predicted_targets_dic['c_o_ratio'], my_list_c_o)
    met0, met1 = find_nearest_top_bottom(predicted_targets_dic['metallicity'], my_list_met)
    T0, T1 = find_nearest_top_bottom(predicted_targets_dic['temperature'], my_list_T)

    filter_params = {'gravity': (g0, g1),
                     'temperature': (T0, T1),
                     'c_o_ratio': (co0, co1),
                     'metallicity': (met0, met1)}

    df_to_interpolate = filter_dataset_range(dataset, filter_params).reset_index(drop=True)

    my_list_g = list(df_to_interpolate['gravity'].sort_values().unique())
    my_list_met = list(df_to_interpolate['metallicity'].sort_values().unique())
    my_list_c_o = list(df_to_interpolate['c_o_ratio'].sort_values().unique())
    my_list_T = list(df_to_interpolate['temperature'].sort_values().unique())

    if print_results_:
        print(df_to_interpolate)
        print(my_list_g, my_list_c_o, my_list_T, my_list_met)

    df_interpolated_ = pd.DataFrame(columns=df_to_interpolate.drop(
        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity']).columns)

    df_interpolated_all = pd.DataFrame(columns=df_to_interpolate.columns)

    for temp in my_list_T:
        for grav in my_list_g:
            for met in my_list_met:
                for c_o in range(0, len(my_list_c_o) - 1):
                    # print(temp, grav, met, c_o)

                    filter_params = {'gravity': (grav, grav),
                                     'temperature': (temp, temp),
                                     'c_o_ratio': (my_list_c_o[c_o], my_list_c_o[c_o + 1]),
                                     'metallicity': (met, met)}

                    df_to_interpolate_ = filter_dataset_range(dataset, filter_params).reset_index(
                        drop=True)  # .drop_duplicates(subset=['gravity', 'temperature', 'c_o_ratio', 'metallicity'])

                    data = df_to_interpolate_.drop(
                        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity'])

                    y = df_to_interpolate_['c_o_ratio'].to_numpy()

                    column_grid = data.columns.to_numpy().astype(float)
                    values = df_to_interpolate_.drop(
                        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity']).to_numpy()

                    interp_func = RegularGridInterpolator((y, column_grid), values)

                    # Define the coordinates for interpolation
                    xi = column_grid  # x-coordinates for interpolation

                    yi = predicted_targets_dic['c_o_ratio']
                    # np.append(np.arange(y[0], y[1], abs(y[1] - y[0])/5, dtype=np.float64),y[1])  # y-coordinates for interpolation
                    xi_mesh, yi_mesh = np.meshgrid(xi, yi, indexing='ij')  # Meshgrid for interpolation

                    # Perform interpolation
                    df_interpolated_ = pd.DataFrame(interp_func((yi_mesh, xi_mesh)).T, columns=data.columns,
                                                    dtype=np.float64)
                    df_interpolated_['c_o_ratio'] = yi_mesh[0]
                    df_interpolated_['temperature'] = temp
                    df_interpolated_['metallicity'] = met
                    df_interpolated_['gravity'] = grav
                    # df_interpolated_['is_augmented'] = 'no'

                    df_interpolated_all = pd.concat([df_interpolated_, df_interpolated_all], ignore_index=True)

    # ***************************************************************************************

    my_list_g = list(df_interpolated_all['gravity'].sort_values().unique())
    my_list_met = list(df_interpolated_all['metallicity'].sort_values().unique())
    my_list_c_o = list(df_interpolated_all['c_o_ratio'].sort_values().unique())
    my_list_T = list(df_interpolated_all['temperature'].sort_values().unique())

    df_interpolated_all2 = df_interpolated_all
    df_interpolated_all2.drop_duplicates(inplace=True)

    df_interpolated_ = pd.DataFrame(columns=df_interpolated_all2.drop(
        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity',]).columns)

    for c_o in my_list_c_o:
        for temp in my_list_T:
            for grav in my_list_g:
                for met in range(0, len(my_list_met) - 1):

                    filter_params = {'gravity': (grav, grav),
                                     'temperature': (temp, temp),
                                     'c_o_ratio': (c_o, c_o),
                                     'metallicity': (my_list_met[met], my_list_met[met + 1])}

                    df_to_interpolate_ = filter_dataset_range(df_interpolated_all2, filter_params).reset_index(
                        drop=True)  # .drop_duplicates(subset=['gravity', 'temperature', 'c_o_ratio', 'metallicity'])

                    data = df_to_interpolate_.drop(
                        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', ])

                    y = df_to_interpolate_['metallicity'].to_numpy()

                    column_grid = data.columns.to_numpy().astype(float)
                    values = df_to_interpolate_.drop(
                        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', ]).to_numpy()

                    interp_func = RegularGridInterpolator((y, column_grid), values)

                    # Define the coordinates for interpolation
                    xi = column_grid  # x-coordinates for interpolation

                    yi = predicted_targets_dic['metallicity']
                    xi_mesh, yi_mesh = np.meshgrid(xi, yi, indexing='ij')  # Meshgrid for interpolation

                    # Perform interpolation
                    df_interpolated_ = pd.DataFrame(interp_func((yi_mesh, xi_mesh)).T, columns=data.columns,
                                                    dtype=np.float64)
                    df_interpolated_['metallicity'] = yi_mesh[0]
                    df_interpolated_['temperature'] = temp
                    df_interpolated_['c_o_ratio'] = c_o
                    df_interpolated_['gravity'] = grav
                    # df_interpolated_['is_augmented'] = 'no'

                    df_interpolated_all = pd.concat([df_interpolated_, df_interpolated_all], ignore_index=True)


    # ************************************************************************************

    my_list_g = list(df_interpolated_all['gravity'].sort_values().unique())
    my_list_met = list(df_interpolated_all['metallicity'].sort_values().unique())
    my_list_c_o = list(df_interpolated_all['c_o_ratio'].sort_values().unique())
    my_list_T = list(df_interpolated_all['temperature'].sort_values().unique())

    if print_results_:
        print(my_list_g, my_list_c_o, my_list_T, my_list_met)

    df_interpolated_all2 = df_interpolated_all
    df_interpolated_all2.drop_duplicates(inplace=True)

    df_interpolated_ = pd.DataFrame(columns=df_interpolated_all2.drop(
        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', ]).columns)

    for c_o in my_list_c_o:
        for met in my_list_met:
            for grav in my_list_g:
                for temp in range(0, len(my_list_T) - 1):
                    # print(temp, grav, met, c_o)

                    filter_params = {'gravity': (grav, grav),
                                     'temperature': (my_list_T[temp], my_list_T[temp + 1]),
                                     'c_o_ratio': (c_o, c_o),
                                     'metallicity': (met, met)}

                    df_to_interpolate_ = filter_dataset_range(df_interpolated_all2, filter_params).reset_index(
                        drop=True)  # .drop_duplicates(subset=['gravity', 'temperature', 'c_o_ratio', 'metallicity'])

                    # print(df_to_interpolate)
                    data = df_to_interpolate_.drop(
                        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', ])

                    y = df_to_interpolate_['temperature'].to_numpy()
                    # print(y)

                    column_grid = data.columns.to_numpy().astype(float)
                    values = df_to_interpolate_.drop(
                        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', ]).to_numpy()

                    interp_func = RegularGridInterpolator((y, column_grid), values)

                    # Define the coordinates for interpolation
                    xi = column_grid  # x-coordinates for interpolation

                    yi = predicted_targets_dic['temperature']

                    xi_mesh, yi_mesh = np.meshgrid(xi, yi, indexing='ij')  # Meshgrid for interpolation

                    # Perform interpolation
                    df_interpolated_ = pd.DataFrame(interp_func((yi_mesh, xi_mesh)).T, columns=data.columns,
                                                    dtype=np.float64)
                    df_interpolated_['temperature'] = yi_mesh[0]
                    df_interpolated_['metallicity'] = met
                    df_interpolated_['c_o_ratio'] = c_o
                    df_interpolated_['gravity'] = grav
                    # df_interpolated_['is_augmented'] = 'no'

                    df_interpolated_all = pd.concat([df_interpolated_, df_interpolated_all], ignore_index=True)

    # ******************************************************************************************

    my_list_g = list(df_interpolated_all['gravity'].sort_values().unique())
    my_list_met = list(df_interpolated_all['metallicity'].sort_values().unique())
    my_list_c_o = list(df_interpolated_all['c_o_ratio'].sort_values().unique())
    my_list_T = list(df_interpolated_all['temperature'].sort_values().unique())

    df_interpolated_all2 = df_interpolated_all
    df_interpolated_all2.drop_duplicates(inplace=True)

    df_interpolated_ = pd.DataFrame(columns=df_interpolated_all2.drop(
        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', ]).columns)

    for c_o in my_list_c_o:
        for met in my_list_met:
            for temp in my_list_T:
                for grav in range(0, len(my_list_g) - 1):
                    # print(temp, grav, met, c_o)

                    filter_params = {'gravity': (my_list_g[grav], my_list_g[grav + 1]),
                                     'temperature': (temp, temp),
                                     'c_o_ratio': (c_o, c_o),
                                     'metallicity': (met, met)}

                    df_to_interpolate_ = filter_dataset_range(df_interpolated_all2, filter_params).reset_index(
                        drop=True)  # .drop_duplicates(subset=['gravity', 'temperature', 'c_o_ratio', 'metallicity'])

                    data = df_to_interpolate_.drop(
                        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', ])

                    y = df_to_interpolate_['gravity'].to_numpy()

                    column_grid = data.columns.to_numpy().astype(float)
                    values = df_to_interpolate_.drop(
                        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', ]).to_numpy()

                    interp_func = RegularGridInterpolator((y, column_grid), values)

                    # Define the coordinates for interpolation
                    xi = column_grid  # x-coordinates for interpolation

                    yi = predicted_targets_dic['gravity']

                    xi_mesh, yi_mesh = np.meshgrid(xi, yi, indexing='ij')  # Meshgrid for interpolation

                    # Perform interpolation
                    df_interpolated_ = pd.DataFrame(interp_func((yi_mesh, xi_mesh)).T, columns=data.columns,
                                                    dtype=np.float64)
                    df_interpolated_['gravity'] = yi_mesh[0]
                    df_interpolated_['metallicity'] = met
                    df_interpolated_['c_o_ratio'] = c_o
                    df_interpolated_['temperature'] = temp
                    # df_interpolated_['is_augmented'] = 'no'

                    df_interpolated_all = pd.concat([df_interpolated_, df_interpolated_all], ignore_index=True)


    df_interpolated_all.drop_duplicates(inplace=True)

    df_interpolated_final = df_interpolated_all[
        (df_interpolated_all['temperature'] == predicted_targets_dic['temperature']) &
        (df_interpolated_all['c_o_ratio'] == predicted_targets_dic['c_o_ratio']) &
        (df_interpolated_all['metallicity'] == predicted_targets_dic['metallicity']) &
        (df_interpolated_all['gravity'] == predicted_targets_dic['gravity']) #&
        # (df_interpolated_all['is_augmented'] == 'no')
    ]

    return df_interpolated_final


# def filter_dataframe(training_datasets, predicted_targets_dic):
def filter_dataframe(training_datasets, predicted_targets_dic):
    nearest_value_list = []
    filtered_df = training_datasets.copy()

    # Check if the values exist in the respective columns
    for col, value in predicted_targets_dic.items():
        if value not in filtered_df[col].values:
            nearest_value = filtered_df[col].values[np.argmin(np.abs(filtered_df[col].values - value))]
            nearest_value_list.append(nearest_value)
            filtered_df = filtered_df[filtered_df[col] == nearest_value]

    return nearest_value_list, filtered_df


def plot_boxplot_hist(data,
                 x_label,
                 xy_loc):

    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

    sns.histplot(data, ax=ax_hist, kde=True, stat='probability')
    sns.boxplot(x = data, ax=ax_box, showmeans=True, meanline = True,
                meanprops={"marker": "|",
                           "markeredgecolor": "white",
                           "markersize": "30",
                            }
                       )

    fig.set_figheight(3)
    fig.set_figwidth(3)

    ax_box.set(xlabel='')
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    ax_box.set_yticks([])

    mean = np.round(np.mean(data),2)
    std = np.round(np.std(data),2)
    plt.annotate(f'{x_label}='+str(np.round(mean,2))+'$\pm$'+str(np.round(std,2)), fontsize=11,
                 xy=(xy_loc[0], xy_loc[1]), xycoords='axes fraction')

    plt.xlabel(x_label, fontsize = 12)
    if x_label == 'C/O':
        x_label = 'c_o_ratio'
    if x_label == '[M/H]':
        x_label = 'metallicity'
    # plt.savefig(f'../outputs/figures/boxplot_hist_{x_label}.pdf', format='pdf')

    plt.show()


def plot_spectra_errorbar(object_name,
                          x_obs,
                          y_obs,
                          y_obs_err,
                          y_label = "Flux (F𝜈) [erg/s/cm2/Hz]",
                          title_label = None,
                          data_type='x_y_yerr'):
    # Create the figure
    p = figure(title=f"{object_name}: Calibrated Observational Spectra" if title_label is None else title_label,
               x_axis_label="Wavelength [𝜇m]",
               y_axis_label=y_label,
               width=800, height=300,
               y_axis_type="log",
               tools="pan,wheel_zoom,box_zoom,reset")



    # Add the scatter plot
    p.scatter(x_obs, y_obs,  size=4, fill_color='green', line_color=None, line_alpha=0.2,
              legend_label=f"{object_name}: Observational data")

    if data_type == 'x_y_yerr':
        # Define maximum error threshold as a percentage of y-value
        max_error_threshold = 0.8

        # Calculate adjusted error bar coordinates
        upper = np.minimum(y_obs + y_obs_err, y_obs + y_obs * max_error_threshold)
        lower = np.maximum(y_obs - y_obs_err, y_obs - y_obs * max_error_threshold)# Sample data
        p.segment(x0=x_obs, y0=lower, x1=x_obs, y1=upper,
                  color='gray', line_alpha=0.7)

    # Increase size of x and y ticks
    p.title.text_font_size = '12pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.xaxis.axis_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'
    p.yaxis.axis_label_text_font_size = '12pt'

    # Show the plot
    output_notebook()
    show(p)


def replace_zeros_with_mean(dataframe_col):
    """
    Replace zero values in a DataFrame column with the mean of their non-zero neighbors.

    Parameters
    ----------
    dataframe_col : pandas.Series
        A pandas Series representing the column of a DataFrame.

    Returns
    -------
    pandas.Series
        The updated pandas Series with zero values replaced by the mean of non-zero neighbors.
    """

    df_col = dataframe_col.copy()
    zero_indices = np.where(df_col.values <= 0)
    non_zero_indices = np.where(df_col.values > 0)

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
        return dataframe_col



def plot_pred_vs_obs_errorbar(object_name,
                                       x_obs,
                                       y_obs,
                                       y_obs_error,
                                       training_dataset,
                                       x_pred,
                                       predicted_targets_dic,
                                       __print_results__= False):
    """
    Plot predicted spectra along with observed spectra and error bars.

    Parameters
    ----------
    object_name : str
        Name of the object being plotted.
    x_obs : list
        List of x-axis values (wavelengths) for the observed spectra.
    y_obs : list
        List of y-axis values (observed feature values) for the observed spectra.
    y_obs_error : list
        List of error values corresponding to the observed feature values.
    training_datasets : list, optional
        Training (or synthetic) datasets used for training the models. Default is None.
    predicted_targets_dic : dict, optional
        Dictionary of predicted targets. Default is None.
    __print_results__ : bool
        True or False.
    """

    # Define maximum error threshold as a percentage of y-value
    max_error_threshold = 0.8


    # Calculate adjusted error bar coordinates
    upper = np.minimum(y_obs + y_obs_error, y_obs + y_obs * max_error_threshold)
    lower = np.maximum(y_obs - y_obs_error, y_obs - y_obs * max_error_threshold)# Sample data


    # Create a ColumnDataSource to store the data
    source = ColumnDataSource(data=dict(x=x_obs, y=y_obs, upper=upper, lower=lower))

    # Create the Observational figure
    p = figure(title=f"{object_name}: Observational vs. ML Predicted Spectra",
               x_axis_label="Wavelength [𝜇m]",
               y_axis_label="TOA Flux (F𝜈) [erg/s/cm2/Hz]",
               width=800, height=300,
               y_axis_type="log",
               tools="pan,wheel_zoom,box_zoom,reset")

    # Add the scatter plot
    p.scatter('x', 'y', source=source, size=4, fill_color='green', line_color=None, line_alpha=0.2, legend_label=f"{object_name}: Observational data")

    # Add the error bars using segment
    p.segment(x0='x', y0='lower', x1='x', y1='upper', source=source, color='gray', line_alpha=0.7)

    # Create the Predicted figure
    ypred = list(predicted_targets_dic.values())

    filtered_df = interpolate_df(dataset=training_dataset,
                                 predicted_targets_dic=predicted_targets_dic, print_results_=False)

    if __print_results__:
        print('------------- Interpolated spectrum based on the ML predicted targets --------------')
        print(filtered_df)

    # Add the scatter plot
    p.line(
        x=x_pred['wl'],
        y=filtered_df.drop(columns=['gravity', 'c_o_ratio', 'metallicity', 'temperature',]).values[0],
        line_width=1,
        legend_label='ML Predicted:' + ', '.join([['log𝑔= ', 'C/O= ', '[M/H]= ', 'T= '][i] + str(np.round(ypred[i], 2)) for i in range(4)])
    )

    # Increase size of x and y ticks
    p.title.text_font_size = '12pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.xaxis.axis_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'
    p.yaxis.axis_label_text_font_size = '12pt'

    p.legend.location = "bottom_left"
    p.legend.background_fill_color = 'white'
    p.legend.background_fill_alpha = 0.5


    show(p)

def plot_pred_vs_obs_errorbar_stat_bokeh(  stat_df,
                                confidence_level,
                                object_name,
                                x_obs,
                                y_obs,
                                y_obs_err,
                                training_datasets,
                                x_pred,
                                predicted_targets_dic,
                                radius,
                                __print_results__ = False):
    """
    Plot observed spectra with error bars and predicted spectra with confidence intervals.

    Parameters
    ----------
    stat_df : DataFrame
        DataFrame containing the calculated statistics.
    confidence_level : float
        Confidence level for the confidence intervals.
    object_name : str
        Name of the object being plotted.
    x_obs : list
        List of x-axis values for the observed spectra.
    y_obs : list
        List of y-axis values for the observed spectra.
    y_obs_err : list
        List of error values corresponding to the observed spectra.
    training_datasets : optional
        Training datasets used for prediction. Default is None.
    predicted_targets_dic : optional
        Dictionary of predicted targets. Default is None.
    # bd_object_class : optional
    #     Object class. Default is None.
    __print_results__ : bool
        True or False.
    """

    chi2_stat, p_value = chi_square_test(
                            x_obs=x_obs,
                            y_obs=y_obs,
                            yerr_obs=y_obs_err,

                            x_pre=stat_df['wl'][::-1],
                            y_pre=stat_df['mean'],
                            yerr_pre=stat_df['std_values'],
                            radius=radius,
                            __plot_results__=False,
                            __print_results__=True)

    if __print_results__:
        print('*'*10+ ' Predicted Targets dic ' + '*'*10 )
        print(predicted_targets_dic)

    # Create a figure


    # Create ML figure
    p = figure(
        title=object_name+': Observational vs. ML Predicted Spectra'+' [𝛘2='+str(chi2_stat)+', p-value='+ str(p_value)+']',
        x_axis_label='Wavelength [μm]',
        y_axis_label='Absolute Flux (F𝜈) [erg/s/cm2/Hz]',
        y_axis_type="log",
        width=1000,
        height=400
    )

    # Create the Observationa  figure * * * * * * * * * * * * * * * * *
    max_error_threshold = 0.8

    # Calculate adjusted error bar coordinates
    upper = np.minimum(y_obs + y_obs_err, y_obs + y_obs * max_error_threshold)
    lower = np.maximum(y_obs - y_obs_err, y_obs - y_obs * max_error_threshold)# Sample data

    # Create a ColumnDataSource to store the data
    source = ColumnDataSource(data=dict(x=x_obs, y=y_obs, upper=upper, lower=lower))

    # Add the scatter plot
    p.scatter('x', 'y', source=source, size=4, fill_color='green', line_color=None, line_alpha=0.2,
              legend_label=f"{object_name}: Observational data")

    # Add the error bars using segment
    p.segment(x0='x', y0='lower', x1='x', y1='upper', source=source, color='gray', line_alpha=0.7)


    # Create the ML Predicted figure * * * * * * * * * * * * * * * * *



    p.line(x=stat_df['wl'][::-1],
           y=stat_df['mean'],
           color='blue', line_width=2,
           legend_label='ML Predicted:' + ', '.join(
               [['log𝑔= ', 'C/O= ', '[M/H]= ', 'T= '][i] + str(np.round(list(predicted_targets_dic.values())[i], 2)) for i in range(4)])+
                ', R='+str(np.round(radius,2))+' Rjup'
          )

    # Plot the shaded regions for confidence intervals
    p.varea(
        x=stat_df['wl'][::-1],
        y1=stat_df['confidence_level_lower'],
        y2=stat_df['confidence_level_upper'],
        fill_color='red',
        fill_alpha=0.8,
        legend_label='Confidence Level: {}%'.format(confidence_level)
    )

    # Plot the shaded regions for 1 sigma
    p.varea(
        x=stat_df['wl'][::-1],
        y1=stat_df['mean'] - stat_df['std_values'],
        y2=stat_df['mean'] + stat_df['std_values'],
        fill_color='green',
        fill_alpha=0.4,
        legend_label='1σ'
    )






    # Customize the plot
    p.title.text_font_size = '12pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.xaxis.axis_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'
    p.yaxis.axis_label_text_font_size = '12pt'
    p.legend.location = "bottom_left"
    p.legend.background_fill_color = 'white'
    p.legend.background_fill_alpha = 0.5
    p.legend.click_policy = 'hide'

    # Print the results if specified
    if __print_results__:
        print("Printing results:")
        print(stat_df.head(5))

    show(p)


def calculate_confidence_intervals_std_df(dataset_df,
                                          __print_results__ = False,
                                          __plot_calculate_confidence_intervals_std_df__ = False
                                         ):
    """
    Calculate confidence intervals and other statistics for a DataFrame.

    Parameters
    ----------
    dataset_df : DataFrame
        The input DataFrame containing the data.
    __print_results__ : bool
        True or False.
    __plot_calculate_confidence_intervals_std_df__ : bool
        True or False.

    Returns
    -------
    DataFrame
        A DataFrame with the calculated statistics.
    """

    # Copy the dataset to avoid modifying the original DataFrame
    df3 = dataset_df.copy()

    confidence_level = 0.95  # Confidence level (e.g., 95% confidence)

    # Calculate the sample size
    n = len(df3)

    # Calculate the mean and standard deviation for each column
    mean_values = df3.mean()
    std_values = df3.std()

    # Calculate the standard error
    se_values = std_values / np.sqrt(n)

    # Calculate the t-value for the desired confidence level and degrees of freedom
    t_value = stats.t.ppf((1 + confidence_level) / 2, df = n - 1)

    # Calculate the confidence interval for each column
    stat_df = pd.DataFrame(columns=['confidence_level_lower', 'confidence_level_upper'], index=None)
    stat_df = stat_df.astype(np.float64)

    for column in df3.columns:
        lower_bound = mean_values[column] - t_value * se_values[column]
        upper_bound = mean_values[column] + t_value * se_values[column]
        stat_df.loc[column] = [lower_bound, upper_bound]

    # Add additional columns to the DataFrame
    stat_df['mean'] = mean_values
    stat_df['std_values'] = std_values
    stat_df['wl'] = np.float64(df3.columns)

    if __print_results__:
        print(stat_df.head(5))

    if __plot_calculate_confidence_intervals_std_df__:
        # Plot the results
        x = np.round(df3.columns, 2)
        p = figure(
            title='Mean with Confidence Intervals',
            x_axis_label='Wavelength [μm]',
            y_axis_label='Flux (F𝜈) [erg/s/cm2/Hz]',
            y_axis_type="log",
            width=1000,
            height=400
        )

        p.line(x = stat_df['wl'][::-1],
               y = stat_df['mean'],
               color = 'blue',
               line_width = 2,
               legend_label='Mean')#+', '.join([['log𝑔= ','C/O= ', '[M/H]= ', 'T= '][i]+str(np.round(ypred[i],2)) for i in  range(4)])))

        p.varea(
            x  = stat_df['wl'][::-1],
            y1 = stat_df['confidence_level_lower'],
            y2 = stat_df['confidence_level_upper'],
            fill_color = 'red',
            fill_alpha=0.8,
            legend_label='Confidence Level: {}%'.format(confidence_level)
        )

        p.varea(
            x  = stat_df['wl'][::-1],
            y1 = stat_df['mean'] - stat_df['std_values'],
            y2 = stat_df['mean'] + stat_df['std_values'],
            fill_color = 'green',
            fill_alpha = 0.4,
            legend_label = '1σ'
        )

        p.legend.click_policy = 'hide'

        show(p)

    return stat_df


def plot_with_errorbars(x_obs, y_obs, err_obs, x_pre, y_pre, err_pre, title="Data with Error Bars"):
    """
    Create a Bokeh plot with custom error bars for two datasets.

    Parameters
    ----------
    x_obs : array
        X-axis values for observed dataset.
    y_obs : array
        Y-axis values for observed dataset.
    err_obs : array
        Error bars for observed dataset (positive values).
    x_pre : array
        X-axis values for predicted dataset.
    y_pre : array
        Y-axis values for predicted dataset.
    err_pre : array
        Error bars for predicted dataset (positive values).
    title : str
        Title of the plot (default is "Data with Error Bars").

    Returns
    -------
    None
        (Displays the plot).
    """

    # Calculate upper and lower error bars for observed dataset
    upper_err_obs = [y_i + err for y_i, err in zip(y_obs, err_obs)]
    lower_err_obs = [y_i - err for y_i, err in zip(y_obs, err_obs)]

    # Calculate upper and lower error bars for predicted dataset
    upper_err_pre = [y_i + err for y_i, err in zip(y_pre, err_pre)]
    lower_err_pre = [y_i - err for y_i, err in zip(y_pre, err_pre)]

    # Create Bokeh ColumnDataSources for both datasets
    source_obs = ColumnDataSource(data=dict(x_obs=x_obs, y_obs=y_obs, upper_err_obs=upper_err_obs, lower_err_obs=lower_err_obs))
    source_pre = ColumnDataSource(data=dict(x_pre=x_pre, y_pre=y_pre, upper_err_pre=upper_err_pre, lower_err_pre=lower_err_pre))

    p = figure(
        x_axis_label='Wavelength [𝜇m]',
        y_axis_label='Flux (F𝜈) [erg/s/cm2/Hz]',
        width=800, height=300,
        y_axis_type='log',
        title=title
    )

    # Plot data points for observed dataset
    p.circle(x='x_obs', y='y_obs', source=source_obs, size=3, color="blue", legend_label="Observed")

    # Plot custom error bars for observed dataset using the segment glyph
    p.segment(x0='x_obs', y0='lower_err_obs', x1='x_obs', y1='upper_err_obs', line_color="grey", source=source_obs)

    # Plot data points for predicted dataset
    p.square(x='x_pre', y='y_pre', source=source_pre, size=3, color="red", legend_label="Predicted")

    # Plot custom error bars for predicted dataset using the segment glyph
    p.segment(x0='x_pre', y0='lower_err_pre', x1='x_pre', y1='upper_err_pre', line_color="grey", source=source_pre)

    # Add legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    # Increase size of x and y ticks
    p.title.text_font_size = '12pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.xaxis.axis_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'
    p.yaxis.axis_label_text_font_size = '12pt'


    p.legend.location = "top_right"
    p.legend.background_fill_color = 'white'
    p.legend.background_fill_alpha = 0.5

    # Show the plot
    output_notebook()
    show(p)


def chi_square_test(x_obs, y_obs, yerr_obs,
                    x_pre, y_pre, yerr_pre,
                    radius,
                    __plot_results__ = False,
                    __print_results__ = True):
    """
    Perform the chi-square test to evaluate the similarity between two datasets with error bars.

    Parameters
    ----------
    x_obs : array
        The x-coordinates of the observed dataset.
    y_obs : array
        The y-coordinates of the observed dataset.
    yerr_obs : array
        The error bars associated with the observed dataset.
    x_pre : array
        The x-coordinates of the predicted dataset.
    y_pre : array
        The y-coordinates of the predicted dataset.
    yerr_pre : array
        The error bars associated with the predicted dataset.
    radius : float
        The radius value for comparison of points between datasets.
    __plot_results__ : bool, optional
        If True, plot the results of the chi-square test. Defaults to False.
    __print_results__ : bool, optional
        If True, print the results of the chi-square test. Defaults to True.

    Returns
    -------
    float
        The chi-square test statistic.
    float
        The p-value.

    Raises
    ------
    ValueError
        If the lengths of the datasets or error bars are not equal.
    """

    indices_to_remove = np.where(np.isnan(y_obs))[0]

    # Create a boolean mask with True for elements to keep, and False for elements to remove
    mask = np.ones(len(y_obs), dtype=bool)
    mask[indices_to_remove] = False
    y_obs = y_obs[mask]
    yerr_obs = yerr_obs[mask]
    x_obs = x_obs[mask]

    # Convert input to NumPy arrays for easier calculations
    data1 = np.asarray(y_obs)
    data2 = np.asarray(y_pre)
    error1 = np.asarray(yerr_obs)
    error2 = np.asarray(yerr_pre)

    num_points = len(x_pre)
    # print(y_obs)

    # Interpolate datasets if they have different lengths
    if len(data1) != len(data2):
        f1 = interp1d(x_obs, data1, kind='cubic', fill_value='extrapolate') #interp1d(x1, data1, kind='quadratic')
        f2 = interp1d(x_pre, data2, kind='cubic', fill_value='extrapolate')
        data1 = f1(x_pre)
        data2 = f2(x_pre)

        f_error1 = interp1d(x_obs, error1, kind='cubic', fill_value='extrapolate')
        f_error2 = interp1d(x_pre, error2, kind='cubic', fill_value='extrapolate')
        error1 = f_error1(x_pre)
        error2 = f_error2(x_pre)


    # Calculate the chi-square test statistic
    chi2_stat = np.round( np.sum(((data1 - data2) / np.sqrt(error1**2 + error2**2))**2), 2)
    # print(data1,data2)

    # Calculate the degrees of freedom
    degrees_of_freedom = len(data1) - 1

    # Calculate the p-value using the chi-square distribution
    p_value = "{:.2e}".format( 1.0 - chi2.cdf(chi2_stat, degrees_of_freedom) )
    # p_value = '{:.2e}'.p_value

    if __plot_results__:
        plot_with_errorbars(x_pre, data1, error1,
                            x_pre, data2, error2,
                            title=f"Radius={'{:.2f}'.format(radius)} R_Jup:  𝛘2={chi2_stat}, p-value={p_value}")

    if __print_results__:
        print( f"Radius = {'{:.2f}'.format(radius)} R_Jup:  𝛘2 = {chi2_stat}, p-value = {p_value}")


    return chi2_stat, p_value


def plot_chi_square_p_value(radius, chi_square_list, p_value_list,
                            ):
    """
    Plot two lines on the same plot with twin y-axis.

    Parameters
    ----------
    radius : array
        The x-axis values.
    chi_square_list : array
        The y-axis values for the first line.
    p_value_list : array
        The y-axis values for the second line.

    Returns
    -------
    None
        (displays the plot).
    """

    # Create a ColumnDataSource to hold the data
    source = ColumnDataSource(data=dict(radius=radius, chi_square_list=chi_square_list, p_value_list=p_value_list))

    # Create the Bokeh figure
    fig = figure(width=800, height=400, title="Chi-square and p-value", x_axis_label="Radius [R_Jup]",
                 y_axis_label="Statistic Test Metric",
                 y_axis_type="log",
                 y_range=(0.01, max(chi_square_list) + 0.20 * max(chi_square_list)))

    # Plot the first line
    fig.scatter('radius', 'chi_square_list', source=source, color='blue', marker='circle',
                legend_label='𝛘2 value')

    # Plot the second line
    fig.scatter('radius', 'p_value_list', source=source, color='red', marker='circle', legend_label='p-value')

    # Add the horizontal line at y=0.05
    fig.line([min(radius), max(radius)], [0.05, 0.05],
             line_color='black', line_dash='dashed', line_width=2, line_alpha=0.7,
             legend_label='Significance Level (𝛼)=0.05')

    # Add legend
    fig.legend.location = "bottom_left"
    fig.legend.click_policy = "hide"
    fig.legend.background_fill_color = 'white'
    fig.legend.background_fill_alpha = 0.5

    # Increase size of x and y ticks
    fig.title.text_font_size = '12pt'
    fig.xaxis.major_label_text_font_size = '12pt'
    fig.xaxis.axis_label_text_font_size = '12pt'
    fig.yaxis.major_label_text_font_size = '12pt'
    fig.yaxis.axis_label_text_font_size = '12pt'

    # Show the plot
    show(fig)



def find_closest_chi_square(df, chi_square_statistic_list):
    """
    Find the closest chi-square test and p-value for a given degrees of freedom (df).

    Parameters
    ----------
    df : int
        Degrees of freedom for the chi-square test.
    chi_square_statistic_list : list
        List of chi-square test statistics.

    Returns
    -------
    closest_chi_square : float
        The closest chi-square test statistic.
    closest_p_value : float
        The p-value corresponding to the closest chi-square test.
    """

    # Significance level (alpha)
    alpha = 0.05

    # Calculate the critical value for the given significance level and df
    critical_value = chi2.ppf(1 - alpha, df)

    closest_chi_square = None
    closest_p_value = None
    closest_difference = float('inf')

    for chi_square in chi_square_statistic_list:
        # Calculate the p-value
        p_value = 1.0 - chi2.cdf(chi_square, df)

        # Calculate the absolute difference between the chi-square statistic and the critical value
        difference = abs(chi_square - critical_value)

        # Update closest_chi_square and closest_p_value if the current test has a smaller difference
        if difference < closest_difference:
            closest_chi_square = chi_square
            closest_p_value = p_value
            closest_difference = difference

    return closest_chi_square, closest_p_value


def plot_scatter_x_y (x, y,
                        plot_title="Scatter Plot",
                        x_label="X-axis Label",
                        y_label="Y-axis Label",
                        plot_width = 800,
                        plot_height = 400):

    # Create a Bokeh figure
    p = figure(
        title=plot_title,
        x_axis_label=x_label,
        y_axis_label=y_label,
        width=plot_width,  # Width of the plot in pixels
        height=plot_height,  # Height of the plot in pixels
        tools="pan,box_zoom,reset,save",  # Enable interactive tools
    )

    # Add a scatter plot
    p.circle(
        x,
        y,
        size=10,  # Size of the data points
        color="blue",
        legend_label="Data",
        line_color="black",  # Color of the circle border
        line_width=2,  # Width of the circle border line
        fill_alpha=0.6,  # Transparency of the circles (0-1)
    )

    # Customize the appearance
    p.title.text_font_size = "16px"
    p.xaxis.axis_label_text_font_size = "14px"
    p.yaxis.axis_label_text_font_size = "14px"
    p.legend.label_text_font_size = "12px"

    # Show grid lines
    p.grid.visible = True
    p.grid.grid_line_color = "gray"
    p.grid.grid_line_dash = "dotted"
    p.grid.grid_line_alpha = 0.5

    # Show the plot
    show(p)

def plot_filtered_spectra(dataset,
                      filter_bounds,
                      feature_to_plot,
                      title_label,
                      wl_synthetic,
                      output_names,
                      __reference_data__,
                      __save_plots__=False):
    """
    Plot a DataFrame with a single x-axis (using column names) and multiple y-axes.

    Parameters:
        - df (pd.DataFrame): DataFrame containing the data to be plotted.
    """

    filtered_df = dataset.copy()
    for feature, bounds in filter_bounds.items():
        lower_bound, upper_bound = bounds
        filtered_df = filtered_df[(filtered_df[feature] >= lower_bound) & (filtered_df[feature] <= upper_bound)]

#         filtered_df2 = filtered_df.sort_values(feature_to_plot, ascending=False).iloc[::1, 4:-1][::-1]

    filtered_df2 = filtered_df.sort_values(feature_to_plot, ascending=True).drop(columns=output_names)

    fig, ax = plt.subplots(figsize=(12, 4))

    x = filtered_df2.columns
    df_transposed = filtered_df2.T  # Transpose the DataFrame

    # Define a color palette
    num_colors = len(df_transposed.columns)  # Number of colors needed (excluding x-axis)
    colors = sns.color_palette('magma', num_colors)

    for i, col in enumerate(df_transposed.columns):
        # print(col)
        if col != 'x':  # Skip the x-axis column
            ax.semilogy(wl_synthetic, df_transposed[col],
                        # label=data[col][:4].values,
                        color=colors[i], alpha=0.7)

    # print(filtered_data.T[col][:4].values[0])
    ax.set_xlabel('Wavelength [$\mu$m]', fontsize = 12)
    ax.set_ylabel(r'TOA F$_{\nu}^{\rm Syn}$  [erg/cm$^2$/s/Hz]', fontsize = 12)
    dict_features = {'temperature': 'Effective Temperature', 'gravity': 'Gravity', 'metallicity': 'Metallicity',
                     'c_o_ratio': 'Carbon-to-oxygen ratio'}
    ax.set_title(dict_features[feature_to_plot] + " " + title_label, fontsize = 14)
    # ax.legend()

    # Get the minimum and maximum values from the data
    # vmin = df_transposed.values.min()
    # vmax = df_transposed.values.max()

    # Add colorbar
    cmap = sns.color_palette('magma', as_cmap=True)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                                              norm=plt.Normalize(vmin=filter_bounds[feature_to_plot][0],
                                                                 vmax=filter_bounds[feature_to_plot][1])), ax=ax)
    # dict_features2 = {'temperature':'T [K]', 'gravity':'log$g$', 'metallicity':'[M/H]', 'c_o_ratio':'C/O ratio'}
    dict_features = {'temperature': 'T$_{eff}$ [K]', 'gravity': 'log$g$', 'metallicity': '[M/H]', 'c_o_ratio': 'C/O'}
    cbar.set_label(dict_features[feature_to_plot], fontsize = 12)

    if __save_plots__:
        plt.savefig(os.path.join(__reference_data__, 'figures', feature_to_plot + "_training_examples.pdf"), dpi=500,
                    bbox_inches='tight')

    plt.show()


def plot_ML_model_loss(trained_ML_model_history=None, title=None):
    """
    Plot the trained model history for all individual target features
    """

    # history = self.trained_model_history if history is None else history
    # Define the epochs as a list
    epochs = list(range(len(trained_ML_model_history['loss'])))

    # Define colorblind-friendly colors
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']

    # Create a new figure
    p = figure(title=title, width=1000, height=300, y_axis_type='log', x_axis_label='Epochs', y_axis_label='Loss')

    # Add the data lines to the figure with colorblind-friendly colors and increased line width
    p.line(epochs, trained_ML_model_history['loss'], line_color=colors[0], line_dash='solid', line_width=2,
           legend_label='Total loss')
    p.line(epochs, trained_ML_model_history['val_loss'], line_color=colors[0], line_dash='dotted', line_width=2)

    p.line(epochs, trained_ML_model_history['output__gravity_loss'], line_color=colors[1], line_dash='solid', line_width=2,
           legend_label='gravity')
    p.line(epochs, trained_ML_model_history['val_output__gravity_loss'], line_color=colors[1], line_dash='dotted', line_width=2)

    p.line(epochs, trained_ML_model_history['output__c_o_ratio_loss'], line_color=colors[2], line_dash='solid', line_width=2,
           legend_label='c_o_ratio')
    p.line(epochs, trained_ML_model_history['val_output__c_o_ratio_loss'], line_color=colors[2], line_dash='dotted', line_width=2)

    p.line(epochs, trained_ML_model_history['output__metallicity_loss'], line_color=colors[3], line_dash='solid', line_width=2,
           legend_label='metallicity')
    p.line(epochs, trained_ML_model_history['val_output__metallicity_loss'], line_color=colors[3], line_dash='dotted', line_width=2)

    p.line(epochs, trained_ML_model_history['output__temperature_loss'], line_color=colors[4], line_dash='solid', line_width=2,
           legend_label='temperature')
    p.line(epochs, trained_ML_model_history['val_output__temperature_loss'], line_color=colors[4], line_dash='dotted', line_width=2)

    # Increase size of x and y ticks
    p.title.text_font_size = '14pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.xaxis.axis_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'
    p.yaxis.axis_label_text_font_size = '12pt'

    # display legend in top left corner (default is top right corner)
    p.legend.location = "bottom_left"
    p.legend.background_fill_color = 'white'
    p.legend.background_fill_alpha = 0.5

    # Show the plot
    show(p)


def plot_ML_model_loss_plt(trained_ML_model_history=None,
                            title=None,
                            __reference_data__=None,
                            __save_plots__=False):
    """
    Plot the trained model history for all individual target features
    """

    # Define the epochs as a list
    epochs = list(range(len(trained_ML_model_history['loss'])))

    # Define colorblind-friendly colors
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']

    # Set Seaborn style and context

    # Create a new figure
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Huber Loss', fontsize=14)
    ax.set_yscale('log')

    # Add the data lines to the figure with colorblind-friendly colors and increased line width
    ax.plot(epochs, trained_ML_model_history['loss'], color=colors[0], linestyle='-', linewidth=1,
             label='Total loss')
    ax.plot(epochs, trained_ML_model_history['val_loss'], color=colors[0], linestyle=':', linewidth=1)

    ax.plot(epochs, trained_ML_model_history['output__gravity_loss'], color=colors[1], linestyle='-', linewidth=1,
             label=r'$\log g$')
    ax.plot(epochs, trained_ML_model_history['val_output__gravity_loss'], color=colors[1], linestyle=':', linewidth=1)

    ax.plot(epochs, trained_ML_model_history['output__c_o_ratio_loss'], color=colors[2], linestyle='-', linewidth=1,
             label='C/O')
    ax.plot(epochs, trained_ML_model_history['val_output__c_o_ratio_loss'], color=colors[2], linestyle=':', linewidth=1)

    ax.plot(epochs, trained_ML_model_history['output__metallicity_loss'], color=colors[3], linestyle='-', linewidth=1,
             label='[M/H]')
    ax.plot(epochs, trained_ML_model_history['val_output__metallicity_loss'], color=colors[3], linestyle=':', linewidth=1)

    ax.plot(epochs, trained_ML_model_history['output__temperature_loss'], color=colors[4], linestyle='-', linewidth=1,
             label=r'$T_{\rm eff}$')
    ax.plot(epochs, trained_ML_model_history['val_output__temperature_loss'], color=colors[4], linestyle=':', linewidth=1)

    # Increase size of ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Set grid color and linestyle
    ax.grid(which='major', linestyle='--', linewidth=0.5, color='gray', alpha=.9)

    # Enable minor ticks for both x and y axes
    ax.minorticks_on()

    # Set minor ticks format
    ax.tick_params(axis='both', which='minor', direction='out', length=3)
    ax.tick_params(axis='both', which='major', direction='out', length=5)


    # Display legend in top left corner
    ax.legend(loc='lower left', fontsize=12)

    plt.tight_layout()
    # Show the plot
    if __save_plots__:
        plt.savefig(os.path.join(__reference_data__, 'figures', "Trained_CNN_Huber_Loss.pdf"), dpi=500,
                    bbox_inches='tight')

    plt.show()


def plot_boxplot(data,
                 title=None, xlabel='Wavelength [$\mu$m]', ylabel='Scaled Values',
                 xticks_list=None, fig_size=(14, 3),
                 saved_file_name = None,
                 __reference_data__ = None,
                 __save_plots__=False,
                ):
    """
    Make a boxplot with the scaled features.

    Description
    -----------
        - Median: middle quartile marks.
        - Inter-quartile range (The middle “box”): 50% of scores fall within the inter-quartile range.
        - Upper quartile: 75% of the scores fall below the upper quartile.
        - Lower quartile: 25% of scores fall below the lower quartile.
    """

    fig, ax = plt.subplots(figsize=fig_size)
    ax.boxplot(data, sym='')

    if len(data) > 10:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)

    # Increase x and y tick font size
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(which='major', color='grey', linestyle=':', linewidth=0.5)


    # Add custom x-ticks
    # custom_xticks = ['Label 1', 'Label 2', 'Label 3', 'Label 4']
    if xticks_list:
        i = 1
        if len(xticks_list) > 100:
            i = 3
        xtick_positions = range(len(xticks_list))
        ax.set_xticks(xtick_positions[::i])
        ax.set_xticklabels(xticks_list[::i])

    if __save_plots__:
        plt.savefig(os.path.join(__reference_data__, 'figures', saved_file_name+ "_.pdf"), dpi=500,
                    bbox_inches='tight')

    plt.tight_layout()
    plt.show()




def plot_tricontour_chi2_radius(tuned_ML_R_param_df,
                                list_=['temperature', 'gravity', 'metallicity', 'c_o_ratio'],
                                __save_plot__=False):
    plt.figure(figsize=(6, 4))
    for target in list_:
        X = tuned_ML_R_param_df[target]
        Y = tuned_ML_R_param_df['radius']
        Z = tuned_ML_R_param_df['chi_square']

        # Levels for contour lines
        levels = np.linspace(Z.min(), Z.max(), 1001)

        # Set vmin and vmax to customize the color bar range
        Zmin = float(Z.min() // 10 * 10)
        Zmax = float(Z.max() // 100 * 100 + 100)

        contour = plt.tricontour(X, Y, Z,
                                 levels=levels,
                                 cmap='viridis', linestyles='dashed', linewidths=1, vmin=Zmin - 1, vmax=Zmax + 1)

        # Target value
        target_value = 1

        # Calculate Euclidean distances
        distances = np.sqrt((Z - target_value) ** 2)

        # Finding the index of the point with the closest chi-square value to the target
        min_index = np.argmin(distances)
        min_X = X[min_index]
        min_Y = Y[min_index]
        min_Z = Z[min_index]

        # Plotting a star at the minimum chi-square point
        plt.scatter(min_X, min_Y, marker='*', color='red', s=300, zorder=10)

        target_dict = {'temperature': '$T_{eff}$',
                       'gravity': '$\log$g',
                       'metallicity': '[M/H]',
                       'c_o_ratio': 'C/O',
                       }
        plt.xlabel(target_dict[target], fontsize=16)
        plt.ylabel('$R_{Jup}$', fontsize=16)
        plt.xticks(fontsize=14, rotation=45)
        plt.yticks(fontsize=14)
        #         plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2f}'))

        # Adding minor ticks and grid
        plt.minorticks_on()
        plt.grid(which='major', linestyle='--', linewidth='0.1', color='black')
        #         plt.grid(which='minor', linestyle=':', linewidth='0.2', color='gray')

        Rstd = np.round(tuned_ML_R_param_df.describe().loc['std']['radius'], 2)
        Tstd = np.round(tuned_ML_R_param_df.describe().loc['std'][target], 2)
        chistd = np.round(tuned_ML_R_param_df.describe().loc['std']['chi_square'], 2)
        print(Rstd, Tstd, chistd)
        #         plt.suptitle(r'$R_{\mathrm{Jup}}^{\chi^2_\mathrm{min}}$=' + f'{round(min_Y, 2)}$\pm${Rstd}, '
        #                   + target_dict[target] + f'= {round(min_X, 2) }$\pm${Tstd}, $\chi_{{min}}^2$={round(min_Z, 1)}',
        #                                                          y=.95, fontsize=15)
        plt.suptitle(f'$\chi_{{min}}^2$={round(min_Z, 1)}' + r', $R_{\chi^2_\mathrm{min}}$=' + f'{round(min_Y, 2)}, '
                     + target_dict[target] + f'= {round(min_X, 2)}$\pm${Tstd}',
                     y=.95, fontsize=15)

        # Using colorbar to fill the color spectrum
        # cbar = plt.colorbar(contour, label=r'$\chi_r^2$',)
        cbar = plt.colorbar(contour)
        cbar.set_label(r'$\chi_r^2$', fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        # Automatically set better values for colorbar ticks
        #         cbar.locator = MaxNLocator(nbins=9)
        cbar.update_ticks()

        plt.tight_layout()

        if __save_plot__:
            plt.savefig('../../outputs/figures/tuned_bohb_batch32_v3_1000epoch_out10_v2_UsedInPAPER_v5/' +
                        tuned_ML_R_param_df['bd_name'] + '_TunedRadius_' + target
                        + '_counterplot.pdf',
                        format='pdf', bbox_inches='tight')
        plt.show()


def plot_pred_vs_obs_errorbar_stat_matplotlib(  stat_df,
                                confidence_level,
                                object_name,
                                x_obs,
                                y_obs,
                                y_obs_err,
                                training_datasets,
                                x_pred,
                                predicted_targets_dic,
                                radius,
                                __print_results__ = False):
    """
    Plot observed spectra with error bars and predicted spectra with confidence intervals.

    Parameters
    ----------
    stat_df : DataFrame
        DataFrame containing the calculated statistics.
    confidence_level : float
        Confidence level for the confidence intervals.
    object_name : str
        Name of the object being plotted.
    x_obs : list
        List of x-axis values for the observed spectra.
    y_obs : list
        List of y-axis values for the observed spectra.
    y_obs_err : list
        List of error values corresponding to the observed spectra.
    training_datasets : optional
        Training datasets used for prediction. Default is None.
    predicted_targets_dic : optional
        Dictionary of predicted targets. Default is None.
    # bd_object_class : optional
    #     Object class. Default is None.
    __print_results__ : bool
        True or False.
    """

    chi2_stat, p_value = chi_square_test(
                            x_obs=x_obs,
                            y_obs=y_obs,
                            yerr_obs=y_obs_err,

                            x_pre=stat_df['wl'][::-1],
                            y_pre=stat_df['mean'],
                            yerr_pre=stat_df['std_values'],
                            radius=radius,
                            __plot_results__=False,
                            __print_results__=True)

    if __print_results__:
        print('*'*10+ ' Predicted Targets dic ' + '*'*10 )
        print(predicted_targets_dic)

    X = stat_df['wl'][::-1]
    Y = stat_df['mean']
    std = stat_df['std_values']

    # Create the figure and axis
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    ax.errorbar(x_obs, y_obs, yerr=y_obs_err,
                fmt='o', color='blue', alpha=0.8, markersize=2, capsize=3, elinewidth=1, ecolor='gray',label=f"Observational data")

    # Plot data points
    ax.plot(stat_df['wl'][::-1], stat_df['mean'], color='red', label='ML predicted', linewidth=2)

    # Shade the region representing standard deviation
    ax.fill_between(X, Y - std, Y + std, alpha=0.4, color='green', label='1$\sigma$')

    # Set logarithmic scale for y-axis
    ax.set_yscale('log')

    # Set labels and title
    ax.set_xlabel('Wavelength [$\mu$m]',fontsize=14)
    ax.set_ylabel(r'TOA Flux ($F_{\nu}$) [erg/s/cm2/Hz]',fontsize=14)
    ax.set_title(f'{object_name}: Observational vs. ML Predicted Spectra'+' [$\chi^2$='+str(chi2_stat)+']',fontsize=16)

    # Display legend
    ax.legend(loc='lower left',fontsize=12)

    min_mean = np.min(stat_df['mean'])
    max_mean = np.max(stat_df['mean'])
    ax.set_ylim((min_mean * 0.1, max_mean * 2))
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # Customize the plot
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    # export_svg(p, filename=f"../../../outputs/figures/{object_name}_obervational_vs_MLpredicted.svg")
    # plt.savefig(f'../../../outputs/figures/{object_name}_obervational_vs_MLpredicted.pdf', format='pdf')

    plt.show()


def plot_regression_report(trained_ML_model,
                           trained_DataProcessor,
                           Xtrain, Xtest, ytrain, ytest,
                           target_i,
                           xy_top=None, xy_bottom=None,
                           __print_results__=False,
                           __save_plots__ = False):
    """
    Generate a regression report for the trained ML/CNN model.

    Parameters
    -----------
    trained_ML_model : object
        Trained regression model.
    trained_DataProcessor: object
        Trained Data Processor Class
    Xtrain : array
        Training set.
    Xtest : array
        Test set.
    ytrain : array
        Training target set.
    ytest : array
        Test target set.
    target_i : int
        Index of the target variable to analyze.
    xy_top : list, optional
        Coordinates for annotations in the top plot. Defaults to [0.55, 0.85].
    xy_bottom : list, optional
        Coordinates for annotations in the bottom plot. Defaults to [0.05, 0.8].
    __print_results__ : bool, optional
        True or False.
    """

    # Apply the trained ML model on the train set to predict the targets
    if xy_bottom is None:
        xy_bottom = [0.05, 0.8]
    if xy_top is None:
        xy_top = [0.55, 0.85]
    y_pred_train = np.array(trained_ML_model.predict(Xtrain))[:, :, 0].T
    y_pred_train_list = trained_DataProcessor.standardize_y_ColumnWise.inverse_transform(y_pred_train)
    y_pred_train_list[:, 3] = 10 ** y_pred_train_list[:, 3]

    y_act_train_list = trained_DataProcessor.standardize_y_ColumnWise.inverse_transform(ytrain)
    y_act_train_list[:, 3] = 10 ** y_act_train_list[:, 3]

    # Apply the trained ML model on the test set to predict the targets
    y_pred_test = np.array(trained_ML_model.predict(Xtest))[:, :, 0].T
    y_pred_test_list = trained_DataProcessor.standardize_y_ColumnWise.inverse_transform(y_pred_test)
    y_pred_test_list[:, 3] = 10 ** y_pred_test_list[:, 3]

    y_act_test_list = trained_DataProcessor.standardize_y_ColumnWise.inverse_transform(ytest)
    y_act_test_list[:, 3] = 10 ** y_act_test_list[:, 3]

    for i in range(0, target_i):
        y_pred_train = y_pred_train_list[:, i]
        y_act_train = y_act_train_list[:, i]
        y_pred_test = y_pred_test_list[:, i]
        y_act_test = y_act_test_list[:, i]

        # Calculate the residual (Predicted - Actual)
        residual_train_list = y_pred_train - y_act_train
        residual_test_list = y_pred_test - y_act_test

        # Calculate mean and standard deviation for residuals
        mean_test = np.round(np.mean(residual_test_list), 2)
        std_test = np.round(np.std(residual_test_list), 2)
        mean_train = np.round(np.mean(residual_train_list), 2)
        std_train = np.round(np.std(residual_train_list), 2)

        # Calculate skewness for residuals
        skew_test = stats.skew(residual_test_list)
        skew_train = stats.skew(residual_train_list)

        # Calculate R-squared scores
        r2_score_train = r2_score(y_pred_train, y_act_train)
        r2_score_test = r2_score(y_pred_test, y_act_test)

        # Calculate  RMSE scores
        rmse_score_train = np.sqrt(mean_squared_error(y_pred_train, y_act_train))
        rmse_score_test = np.sqrt(mean_squared_error(y_pred_test, y_act_test))

        # Create subplots for histograms and scatter plots
        f, axs = plt.subplots(2, 1, figsize=(5, 5), sharey=False, sharex=False,
                              gridspec_kw=dict(height_ratios=[1, 3]))

        # Turn on minor ticks
        axs[0].minorticks_on()
        axs[1].minorticks_on()

        if __print_results__:
            print('\n\n----------------------- Test ------------------------')
            print('R2: {:2.2f} \t  RMSE: {:2.2f} \t Mean+/-STD: {:2.2f}+/-{:2.2f}'.format(
                r2_score_test, rmse_score_train, mean_test, std_test))

            print('\n----------------------- Train ------------------------')
            print('R2: {:2.2f} \t  RMSE: {:2.2f} \t Mean+/-STD: {:2.2f}+/-{:2.2f}'.format(
                r2_score_train, rmse_score_test, mean_train, std_train))

        # Plot histograms of residuals
        axs[0].set_title(['$\log g$', 'C/O', '[M/H]', '$T_{eff}$'][i], fontsize=14)
        sns.histplot(data=residual_train_list, ax=axs[0], label='train', alpha=0.7, bins=19,
                     log_scale=False, stat='percent', legend=True, linewidth=0)
        sns.histplot(data=residual_test_list, label='test', ax=axs[0], alpha=0.3, bins=19,
                     stat='percent', legend=True, linewidth=0)
        axs[0].set_xlim((-(abs(mean_train) + 3 * std_train), (abs(mean_train) + 3 * std_train)))
        axs[0].set_ylim((1e-1, 100))
        axs[0].set_yscale('log')
        axs[0].set_ylabel('Probability %', fontsize=12)
        axs[0].set_xlabel('Residual', fontsize=12)
        axs[0].grid(which='major', color='grey', linestyle=':', linewidth=0.5)
        axs[1].grid(which='major', color='grey', linestyle=':', linewidth=0.5)

        # Plot scatter figures of predicted vs actual values
        sns.scatterplot(y=y_pred_train, x=y_act_train, label='train', ax=axs[1], alpha=0.7, legend=False)
        sns.scatterplot(y=y_pred_test, x=y_act_test, label='test', ax=axs[1], alpha=0.7, legend=False)
        axs[1].set_ylabel('Predicted value', fontsize=12)
        axs[1].set_xlabel('Actual value', fontsize=12)
        axs[1].xaxis.set_minor_locator(plt.MultipleLocator(100))  # Adjust the step size as needed
        axs[1].yaxis.set_minor_locator(plt.MultipleLocator(100))  # Adjust the step size as needed
        if i < 3:
            axs[1].xaxis.set_minor_locator(plt.MultipleLocator(0.1))  # Adjust the step size as needed
            axs[1].yaxis.set_minor_locator(plt.MultipleLocator(0.1))  # Adjust the step size as needed

        # Increase x and y tick font size
        axs[0].tick_params(axis='both', which='major', labelsize=12)
        axs[1].tick_params(axis='both', which='major', labelsize=12)

        # Add annotations for skewness and R-squared scores
        #         axs[0].annotate(r'$\tilde{\mu}_{{\rm 3, train}}$= ' + f'{np.round(skew_train, 2)}',
        #                         fontsize=11, xy=(xy_top[0], xy_top[1] + 0.08), xycoords='axes fraction')
        #         axs[0].annotate(r'$\tilde{\mu}_{{\rm 3, test}}$ = ' + f'{np.round(skew_test, 2)}',
        #                         fontsize=11, xy=(xy_top[0], xy_top[1] - 0.08), xycoords='axes fraction')
        axs[1].annotate(r'R$^2_{\rm train}$=' + f'{"%0.2f" % r2_score_train}',
                        fontsize=11, xy=(xy_bottom[0], xy_bottom[1] + 0.06), xycoords='axes fraction')
        axs[1].annotate(r'R$^2_{\rm test}$ =' + f'{"%0.2f" % r2_score_test}',
                        fontsize=11, xy=(xy_bottom[0], xy_bottom[1] - 0.06), xycoords='axes fraction')

        axs[1].legend(loc='lower right', fontsize=12)

        plt.tight_layout()
        if __save_plots__:
            target_name = ['Gravity', 'C_O_ratio', 'Metallicity', 'Temperature'][i]
            plt.savefig(f'../../manuscript/2023_ApJ/figures/performance/regression_report_{target_name}_v2.pdf', format='pdf')
        plt.show()
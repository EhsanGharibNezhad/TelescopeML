# Import functions/Classes from other modules ====================
from io_funs import LoadSave
# from predict_observational_dataset_v2 import ProcessObservationalDataset
# from predict_observational_dataset_v3 import ProcessObservationalDataset
from DeepRegTrainer import *


# Import python libraries ========================================
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

from tensorflow.keras.models import save_model
import pickle as pk

# Import BOHB Package ========================================
import logging
logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.examples.commons import MyWorker

from tensorflow.keras.models import load_model


# from bokeh.io import output_notebook
# from bokeh.layouts import row, column
# output_notebook()
# from bokeh.plotting import show,figure
# TOOLTIPS = [
#     ("index", "$index"),
#     ("(x,y)", "($x, $y)"),
# ]


from sklearn.metrics import mean_squared_error, r2_score, make_scorer, explained_variance_score

from scipy import stats
import seaborn as sns


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from scipy.interpolate import RegularGridInterpolator

      

# import pandas as pd
# import numpy as np
from scipy.interpolate import RegularGridInterpolator


import pprint

def print_results_fun(targets, print_title = None):
    
    print('*'*30+'\n')
    
    print(print_title)
    pprint.pprint(targets, indent=4, width=30)
    
    print('*'*30+'\n')

    

def regression_report(trained_model, Xtrain, Xtest, ytrain, ytest, target_i,
                      xy_top=[0.55, 0.85], xy_bottom=[0.05, 0.8], print_results=True):
    """
    Generate a regression report for a trained model.

    Args:
        trained_model (object): Trained regression model.
        Xtrain (numpy.ndarray): Training input data.
        Xtest (numpy.ndarray): Test input data.
        ytrain (numpy.ndarray): Training target data.
        ytest (numpy.ndarray): Test target data.
        target_i (int): Index of the target variable to analyze.
        xy_top (list, optional): Coordinates for annotations in the top plot. Defaults to [0.55, 0.85].
        xy_bottom (list, optional): Coordinates for annotations in the bottom plot. Defaults to [0.05, 0.8].
        print_results (bool, optional): Whether to print the regression results. Defaults to True.
    """

    # Extract predictions from the trained model
    y_pred_train = np.array(trained_model.trained_model.predict(Xtrain))[:, :, 0].T
    y_pred_train_list = trained_model.standardize_y_ColumnWise.inverse_transform(y_pred_train)
    y_pred_train_list[:, 3] = 10 ** y_pred_train_list[:, 3]

    y_act_train_list = trained_model.standardize_y_ColumnWise.inverse_transform(ytrain)
    y_act_train_list[:, 3] = 10 ** y_act_train_list[:, 3]

    y_pred_test = np.array(trained_model.trained_model.predict(Xtest))[:, :, 0].T
    y_pred_test_list = trained_model.standardize_y_ColumnWise.inverse_transform(y_pred_test)
    y_pred_test_list[:, 3] = 10 ** y_pred_test_list[:, 3]

    y_act_test_list = trained_model.standardize_y_ColumnWise.inverse_transform(ytest)
    y_act_test_list[:, 3] = 10 ** y_act_test_list[:, 3]

    for i in range(0, target_i):
        y_pred_train = y_pred_train_list[:, i]
        y_act_train = y_act_train_list[:, i]
        y_pred_test = y_pred_test_list[:, i]
        y_act_test = y_act_test_list[:, i]

        residual_train_list = y_pred_train - y_act_train
        residual_test_list = y_pred_test - y_act_test

        # Create subplots for histograms and scatter plots
        f, axs = plt.subplots(2, 1, figsize=(5, 5), sharey=False, sharex=False,
                              gridspec_kw=dict(height_ratios=[1, 3]))

        # Calculate R-squared scores and RMSE scores
        r2_score_train = r2_score(y_pred_train, y_act_train)
        r2_score_test = r2_score(y_pred_test, y_act_test)

        rmse_score_train = np.sqrt(mean_squared_error(y_pred_train, y_act_train))
        rmse_score_test = np.sqrt(mean_squared_error(y_pred_test, y_act_test))

        # Calculate mean and standard deviation for residuals
        mean_test = np.round(np.mean(residual_test_list), 2)
        std_test = np.round(np.std(residual_test_list), 2)
        mean_train = np.round(np.mean(residual_train_list), 2)
        std_train = np.round(np.std(residual_train_list), 2)

        # Calculate skewness for residuals
        skew_test = stats.skew(residual_test_list)
        skew_train = stats.skew(residual_train_list)

        if print_results:
            print('\n\n----------------------- Test ------------------------')
            print('R2: {:2.2f} \t  RMSE: {:2.2f} \t Mean+/-STD: {:2.2f}+/-{:2.2f}'.format(
                r2_score_test, rmse_score_train, mean_test, std_test))

            print('\n----------------------- Train ------------------------')
            print('R2: {:2.2f} \t  RMSE: {:2.2f} \t Mean+/-STD: {:2.2f}+/-{:2.2f}'.format(
                r2_score_train, rmse_score_test, mean_train, std_train))

        # Plot histograms of residuals
        axs[0].set_title(['Gravity', 'C_O_ratio', 'Metallicity', 'Temperature'][i], fontsize=14)
        sns.histplot(data=residual_train_list, ax=axs[0], label='train', alpha=0.7, bins=19,
                     log_scale=False, stat='percent', legend=True, linewidth=0)
        sns.histplot(data=residual_test_list, label='test', ax=axs[0], alpha=0.3, bins=19,
                     stat='percent', legend=True, linewidth=0)
        axs[0].set_xlim((-(abs(mean_train) + 3 * std_train), (abs(mean_train) + 3 * std_train)))
        axs[0].set_ylim((1e-1, 100))
        axs[0].set_yscale('log')
        axs[0].set_ylabel('Probability %', fontsize=12)

        # Plot scatter plots of predicted vs actual values
        sns.scatterplot(y=y_pred_train, x=y_act_train, label='train', ax=axs[1], alpha=0.7, legend=False)
        sns.scatterplot(y=y_pred_test, x=y_act_test, label='test', ax=axs[1], alpha=0.7, legend=False)
        axs[1].set_ylabel('Predicted value', fontsize=12)
        axs[1].set_xlabel('Actual value', fontsize=12)

        # Add annotations for skewness and R-squared scores
        axs[0].annotate(r'$\tilde{\mu}_{{\rm 3, train}}$= ' + f'{np.round(skew_train, 2)}',
                        fontsize=11, xy=(xy_top[0], xy_top[1] + 0.08), xycoords='axes fraction')
        axs[0].annotate(r'$\tilde{\mu}_{{\rm 3, test}}$ = ' + f'{np.round(skew_test, 2)}',
                        fontsize=11, xy=(xy_top[0], xy_top[1] - 0.08), xycoords='axes fraction')
        axs[1].annotate(r'R$^2_{\rm train}$=' + f'{"%0.2f" % r2_score_train} [{"%0.2f" % abs(mean_train)}$\pm${"%0.2f" % std_train}]',
                        fontsize=11, xy=(xy_bottom[0], xy_bottom[1] + 0.06), xycoords='axes fraction')
        axs[1].annotate(r'R$^2_{\rm test}$ =' + f'{np.round(r2_score_test, 2)} [{"%0.2f" % mean_test}$\pm${"%0.2f" % std_test}]',
                        fontsize=11, xy=(xy_bottom[0], xy_bottom[1] - 0.06), xycoords='axes fraction')

        axs[1].legend(loc='lower right', fontsize=11)

        f.tight_layout()
        plt.show()

        
  

 

def filter_dataset_range(dataset, filter_params):
    filtered_df = dataset.copy()
    
    for param, bounds in filter_params.items():
        lower_bound, upper_bound = bounds
        filtered_df = filtered_df[(filtered_df[param] >= lower_bound) & (filtered_df[param] <= upper_bound)]
        
    return filtered_df

def find_nearest_top_bottom(value, lst):
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
                   print_results_ = False):
    

    
    my_list_g = list(dataset['gravity'].sort_values().unique())
    my_list_met = list(dataset['metallicity'].sort_values().unique())
    my_list_c_o = list(dataset['c_o_ratio'].sort_values().unique())
    my_list_T = list(dataset['temperature'].sort_values().unique())


    g0, g1 = find_nearest_top_bottom(predicted_targets_dic['gravity'], my_list_g)
    co0, co1 = find_nearest_top_bottom(predicted_targets_dic['c_o_ratio'], my_list_c_o)
    met0, met1 = find_nearest_top_bottom(predicted_targets_dic['metallicity'], my_list_met)
    T0, T1 = find_nearest_top_bottom(predicted_targets_dic['temperature'], my_list_T)

    filter_params = {'gravity': (g0 , g1),
                     'temperature': (T0,T1),
                     'c_o_ratio': (co0, co1),
                     'metallicity': (met0, met1)}

    df_to_interpolate = filter_dataset_range(dataset, filter_params).reset_index(drop=True)


    my_list_g = list(df_to_interpolate['gravity'].sort_values().unique())
    my_list_met = list(df_to_interpolate['metallicity'].sort_values().unique())
    my_list_c_o = list(df_to_interpolate['c_o_ratio'].sort_values().unique())
    my_list_T = list(df_to_interpolate['temperature'].sort_values().unique())

    if print_results_:
        print(my_list_g, my_list_c_o, my_list_T, my_list_met)



    df_interpolated_ = pd.DataFrame(columns=df_to_interpolate.drop(
            columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented']).columns)
    df_interpolated_all = []




    for temp in my_list_T: 
        for grav in my_list_g:
            for met in my_list_met:
                for c_o in range(0,len(my_list_c_o)-1):
                    #print(temp, grav, met, c_o)


                    filter_params = {'gravity': (grav, grav),
                     'temperature': (temp, temp),
                     'c_o_ratio': (my_list_c_o[c_o], my_list_c_o[c_o+1]),
                     'metallicity': (met, met)}



                    df_to_interpolate_ = filter_dataset_range(dataset, filter_params).reset_index(
                    drop=True)#.drop_duplicates(subset=['gravity', 'temperature', 'c_o_ratio', 'metallicity'])


                    # display(df_to_interpolate)                                
                    data = df_to_interpolate_.drop(columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented'])



                    y = df_to_interpolate_['c_o_ratio'].to_numpy()

                    column_grid = data.columns.to_numpy().astype(float)
                    values = df_to_interpolate_.drop(columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented']).to_numpy()

                    interp_func = RegularGridInterpolator((y,column_grid),values)

                    # Define the coordinates for interpolation
                    xi = column_grid  # x-coordinates for interpolation

                    yi = predicted_targets_dic['c_o_ratio']
                    #np.append(np.arange(y[0], y[1], abs(y[1] - y[0])/5, dtype=np.float64),y[1])  # y-coordinates for interpolation
                    xi_mesh, yi_mesh = np.meshgrid(xi, yi, indexing='ij')  # Meshgrid for interpolation

                    # Perform interpolation
                    df_interpolated_ = pd.DataFrame(interp_func((yi_mesh, xi_mesh)).T, columns=data.columns, dtype=np.float64)
                    df_interpolated_['c_o_ratio'] = yi_mesh[0]
                    df_interpolated_['temperature'] = temp
                    df_interpolated_['metallicity'] = met
                    df_interpolated_['gravity'] = grav    
                    df_interpolated_['is_augmented'] = 'no'    

                    df_interpolated_all.append(df_interpolated_)


    df_interpolated_all = df_interpolated_.append(df_interpolated_all, ignore_index=True)       


    df_interpolated_.append(df_interpolated_all, ignore_index=True)         


    # ***************************************************************************************

    my_list_g = list(df_interpolated_all['gravity'].sort_values().unique())
    my_list_met = list(df_interpolated_all['metallicity'].sort_values().unique())
    my_list_c_o = list(df_interpolated_all['c_o_ratio'].sort_values().unique())
    my_list_T = list(df_interpolated_all['temperature'].sort_values().unique())

    print(my_list_g, my_list_c_o, my_list_T, my_list_met)


    df_interpolated_all2 = df_interpolated_all
    df_interpolated_all2.drop_duplicates(inplace=True)

    df_interpolated_ = pd.DataFrame(columns=df_interpolated_all2.drop(
            columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented']).columns)
    df_interpolated_all = []



    for c_o in my_list_c_o:
        for temp in my_list_T: 
            for grav in my_list_g:
                for met in range(0,len(my_list_met)-1):
                    #print(temp, grav, met, c_o)


                    filter_params = {'gravity': (grav, grav),
                     'temperature': (temp, temp),
                     'c_o_ratio': (c_o, c_o),
                     'metallicity': (my_list_met[met], my_list_met[met+1] )}



                    df_to_interpolate_ = filter_dataset_range(df_interpolated_all2, filter_params).reset_index(
                    drop=True)#.drop_duplicates(subset=['gravity', 'temperature', 'c_o_ratio', 'metallicity'])

                    # display(df_to_interpolate_)


                    # display(df_to_interpolate)                                
                    data = df_to_interpolate_.drop(columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented'])



                    y = df_to_interpolate_['metallicity'].to_numpy()
                    #print(y)

                    column_grid = data.columns.to_numpy().astype(float)
                    values = df_to_interpolate_.drop(columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented']).to_numpy()

                    interp_func = RegularGridInterpolator((y,column_grid),values)

                    # Define the coordinates for interpolation
                    xi = column_grid  # x-coordinates for interpolation

                    yi = predicted_targets_dic['metallicity'] 
                    #np.append(np.arange(y[0], y[1], abs(y[1] - y[0])/5, dtype=np.float64),y[1])  # y-coordinates for interpolation
                    xi_mesh, yi_mesh = np.meshgrid(xi, yi, indexing='ij')  # Meshgrid for interpolation

                    # Perform interpolation
                    df_interpolated_ = pd.DataFrame(interp_func((yi_mesh, xi_mesh)).T, columns=data.columns, dtype=np.float64)
                    df_interpolated_['metallicity'] = yi_mesh[0]
                    df_interpolated_['temperature'] = temp
                    df_interpolated_['c_o_ratio'] = c_o
                    df_interpolated_['gravity'] = grav    
                    df_interpolated_['is_augmented'] = 'no'    

                    df_interpolated_all.append(df_interpolated_)


    df_interpolated_all = df_interpolated_.append(df_interpolated_all, ignore_index=True)  

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
            columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented']).columns)
    df_interpolated_all = []


    # df_interpolated_all = df_interpolated_all_c_o.append(df_interpolated_all_c_o_list, ignore_index=True)


    for c_o in my_list_c_o:
        for met in my_list_met: 
            for grav in my_list_g:
                for temp in range(0,len(my_list_T)-1):
                    #print(temp, grav, met, c_o)


                    filter_params = {'gravity': (grav, grav),
                     'temperature': (my_list_T[temp], my_list_T[temp+1]),
                     'c_o_ratio': (c_o, c_o),
                     'metallicity': (met, met )}



                    df_to_interpolate_ = filter_dataset_range(df_interpolated_all2, filter_params).reset_index(
                    drop=True)#.drop_duplicates(subset=['gravity', 'temperature', 'c_o_ratio', 'metallicity'])


                    # display(df_to_interpolate)                                
                    data = df_to_interpolate_.drop(columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented'])



                    y = df_to_interpolate_['temperature'].to_numpy()
                    #print(y)

                    column_grid = data.columns.to_numpy().astype(float)
                    values = df_to_interpolate_.drop(columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented']).to_numpy()

                    interp_func = RegularGridInterpolator((y,column_grid),values)

                    # Define the coordinates for interpolation
                    xi = column_grid  # x-coordinates for interpolation

                    yi = predicted_targets_dic['temperature']

                    xi_mesh, yi_mesh = np.meshgrid(xi, yi, indexing='ij')  # Meshgrid for interpolation

                    # Perform interpolation
                    df_interpolated_ = pd.DataFrame(interp_func((yi_mesh, xi_mesh)).T, columns=data.columns, dtype=np.float64)
                    df_interpolated_['temperature'] = yi_mesh[0]
                    df_interpolated_['metallicity'] = met
                    df_interpolated_['c_o_ratio'] = c_o
                    df_interpolated_['gravity'] = grav    
                    df_interpolated_['is_augmented'] = 'no'    

                    df_interpolated_all.append(df_interpolated_)


    df_interpolated_all = df_interpolated_.append(df_interpolated_all, ignore_index=True)  

    # ******************************************************************************************


    my_list_g = list(df_interpolated_all['gravity'].sort_values().unique())
    my_list_met = list(df_interpolated_all['metallicity'].sort_values().unique())
    my_list_c_o = list(df_interpolated_all['c_o_ratio'].sort_values().unique())
    my_list_T = list(df_interpolated_all['temperature'].sort_values().unique())



    df_interpolated_all2 = df_interpolated_all
    df_interpolated_all2.drop_duplicates(inplace=True)


    df_interpolated_ = pd.DataFrame(columns=df_interpolated_all2.drop(
            columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented']).columns)
    df_interpolated_all = []



    for c_o in my_list_c_o:
        for met in my_list_met: 
            for temp in my_list_T:
                for grav in range(0,len(my_list_g)-1):
                    #print(temp, grav, met, c_o)


                    filter_params = {'gravity': (my_list_g[grav], my_list_g[grav+1]),
                     'temperature': (temp, temp),
                     'c_o_ratio': (c_o, c_o),
                     'metallicity': (met, met )}



                    df_to_interpolate_ = filter_dataset_range(df_interpolated_all2, filter_params).reset_index(
                    drop=True)#.drop_duplicates(subset=['gravity', 'temperature', 'c_o_ratio', 'metallicity'])

                    # display(df_to_interpolate_)


                    # display(df_to_interpolate)                                
                    data = df_to_interpolate_.drop(columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented'])



                    y = df_to_interpolate_['gravity'].to_numpy()
                    #print(y)

                    column_grid = data.columns.to_numpy().astype(float)
                    values = df_to_interpolate_.drop(columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented']).to_numpy()

                    interp_func = RegularGridInterpolator((y,column_grid),values)

                    # Define the coordinates for interpolation
                    xi = column_grid  # x-coordinates for interpolation

                    yi = predicted_targets_dic['gravity'] 
                    
                    xi_mesh, yi_mesh = np.meshgrid(xi, yi, indexing='ij')  # Meshgrid for interpolation

                    # Perform interpolation
                    df_interpolated_ = pd.DataFrame(interp_func((yi_mesh, xi_mesh)).T, columns=data.columns, dtype=np.float64)
                    df_interpolated_['gravity'] = yi_mesh[0]
                    df_interpolated_['metallicity'] = met
                    df_interpolated_['c_o_ratio'] = c_o
                    df_interpolated_['temperature'] = temp    
                    df_interpolated_['is_augmented'] = 'no'    

                    df_interpolated_all.append(df_interpolated_)


    df_interpolated_all = df_interpolated_.append(df_interpolated_all, ignore_index=True)  
    df_interpolated_all.drop_duplicates(inplace=True)
    
    return df_interpolated_all


import random
from bokeh.palettes import Magma, Inferno, Plasma, Viridis, Cividis
from bokeh.palettes import  viridis, inferno
       

            
def plot_predicted_vs_observed(training_datasets, 
                               wl,
                               predicted_targets_dic,
                               object_name,
                               bd_object_class,
                               print_results = True,
                              ):
    

    
    ypred = list( predicted_targets_dic.values() )
    
    filtered_df = interpolate_df(dataset=training_datasets, 
                       predicted_targets_dic = predicted_targets_dic,
                       print_results_ = False)

    display(filtered_df)
    
    
    p = figure(
        # title=f'{object_name} [XStand, yStand] Predicted: '+', '.join([['logg= ','C/O= ', 'Met= ', 'T= '][i]+str(np.round(y_pred[0][i],2)) for i in  range(4)]), 
               x_axis_label='Features (Wavelength [𝜇m])', 
               y_axis_label='Flux (F𝜈)',
               width=1000, height=300,
               y_axis_type = 'log')

    # Add the scatter plot

    p.line(x =wl['wl'] , y=filtered_df.drop(columns=['gravity', 'c_o_ratio', 'metallicity', 'temperature','is_augmented']).values[0], 
           line_width = 1,
           legend_label= 'ML Predicted:'+', '.join([['log𝑔= ','C/O= ', '[M/H]= ', 'T= '][i]+str(np.round(ypred[i],2)) for i in  range(4)]))

    if print_results:
        display(bd_object_class.df_flux_object.iloc[:, ::-1])

    p.line(x = wl['wl'] , y = bd_object_class.df_flux_object.iloc[:, ::-1].values[0],
           line_color = 'orange', line_width = 2,
           legend_label='Observational')
    
    p.circle(x = wl['wl'] , y = bd_object_class.df_flux_object.iloc[:, ::-1].values[0],#.iloc[:,4:-1].values[0],
           line_width = 2,
           color='orange'
            )

    # Increase size of x and y ticks
    p.title.text_font_size = '12pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.xaxis.axis_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'
    p.yaxis.axis_label_text_font_size = '12pt'


    p.legend.location = "top_right"
    p.legend.background_fill_color = 'white'
    p.legend.background_fill_alpha = 0.5


    show(p)
    
    
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



import seaborn as sns
def boxplot_hist(data, 
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

    plt.show()
    
    
    
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.io import output_notebook



def plot_spectra_errorbar_old(object_name, 
                          features, 
                          feature_values, 
                          error):
    
    display(error)
    # Calculate the error bar coordinates
    upper = [y_val + err_val for y_val, err_val in zip(feature_values, error)]
    lower = [y_val - err_val for y_val, err_val in zip(feature_values, error)]

    # Create a ColumnDataSource to store the data
    source = ColumnDataSource(data=dict(x=features, y=feature_values, upper=upper, lower=lower))

    # Create the figure
    p = figure(title=f"{object_name}: Calibrated Observational Spectra",
               x_axis_label="Features (Wavelength [𝜇m])",
               y_axis_label="Flux (F𝜈)",
               width=1000, height=300,
               y_axis_type="log",
               tools="pan,wheel_zoom,box_zoom,reset")

    # Increase size of x and y ticks
    p.title.text_font_size = '12pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.xaxis.axis_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'
    p.yaxis.axis_label_text_font_size = '12pt'

    # Add the scatter plot
    p.scatter('x', 'y', source=source, size=4, fill_color='green', line_color=None, line_alpha=0.2, legend_label=f"{object_name}: Observational data")

    # Add the error bars using segment
    p.segment(x0='x', y0='lower', x1='x', y1='upper', source=source, color='gray', line_alpha=0.7)

    # Show the plot
    output_notebook()
    show(p)

    
def plot_spectra_errorbar(object_name, 
                          features, 
                          feature_values, 
                          error):
    
    # Calculate the error bar coordinates
    upper = [y_val + err_val if not np.isnan(err_val) else y_val for y_val, err_val in zip(feature_values, error)]
    lower = [y_val - err_val if not np.isnan(err_val) else y_val for y_val, err_val in zip(feature_values, error)]

    # Create a ColumnDataSource to store the data
    source = ColumnDataSource(data=dict(x=features, y=feature_values, upper=upper, lower=lower))

    # Create the figure
    p = figure(title=f"{object_name}: Calibrated Observational Spectra",
               x_axis_label="Features (Wavelength [𝜇m])",
               y_axis_label="Flux (F𝜈)",
               width=1000, height=300,
               y_axis_type="log",
               tools="pan,wheel_zoom,box_zoom,reset")

    # Increase size of x and y ticks
    p.title.text_font_size = '12pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.xaxis.axis_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'
    p.yaxis.axis_label_text_font_size = '12pt'

    # Add the scatter plot
    p.scatter('x', 'y', source=source, size=4, fill_color='green', line_color=None, line_alpha=0.2, legend_label=f"{object_name}: Observational data")

    # Add the error bars using segment
    p.segment(x0='x', y0='lower', x1='x', y1='upper', source=source, color='gray', line_alpha=0.7)

    # Show the plot
    output_notebook()
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
    
    
    
"""
    
def PredictObsParametersRegression3(object_name, # the name of star, brown dwarf 
                                    dataset, # modeled/synthetic dataset
                                    wl,
                                    train_cnn_regression_class,
                                    predict_from_random_spectra = True,
                                    random_spectra_num=10,
                                    print_results_ = True,
                                    plot_randomly_generated_spectra = True,
                                    print_df_describe = True,
                                    plot_histogram = True,
                                    plot_observational_spectra_errorbar = True,
                                    plot_boxplot_hist = True,
                                    plot_predicted_vs_observed_ = True,
                                    plot_predicted_vs_spectra_errorbar_ = True
                                   ):

    if random_spectra_num < 101:
        color = viridis(random_spectra_num).__iter__()

    
    # Create empty list of spectra and parameters
    spectra_list = []
    param_list = []    


    # load the observational spectra 
    obs_data = pd.read_csv(f'../../datasets/observational_spectra/{object_name}_fluxcal.dat', 
                       delim_whitespace=True, comment='#', names=('wl','F_lambda','F_lambda_error'), 
                       usecols=(0,1,2))#.dropna(inplace=True)

    # Clean the observational spectra: Replace negative fluxes with ZEROs and NAN to zeros
    obs_data['F_lambda']=obs_data['F_lambda'].mask(obs_data['F_lambda'].lt(0),0)
    obs_data['F_lambda'].replace(0, np.nan, inplace=True)
    
    # Interpolate the observational spectra 
    obs_data['F_lambda'].interpolate(inplace=True)



    # if plot_observational_spectra_errorbar_Flam:
    #     plot_spectra_errorbar(object_name, 
    #                           features = obs_data['wl'], 
    #                           feature_values = obs_data['F_lambda'],
    #                           error = obs_data['F_lambda_error'])    
    
    if plot_observational_spectra_errorbar:
        plot_spectra_errorbar(object_name, 
                              features = obs_data['wl'], 
                              feature_values = obs_data['F_lambda'],
                              error = obs_data['F_lambda_error'])    
    
    
    # Instintiate ProcessObservationalDataset class
    bd_object = ProcessObservationalDataset( feature_values_obs = obs_data['F_lambda'].values,
                                            feature_values_obs_err = obs_data['F_lambda_error'].values,
                                             feature_names_obs  = obs_data['wl'].values, 
                                             feature_names_synthetic = wl['wl'].values,
                                             # feature_values_obs_error = fluxcal['F_lambda_error'].values,
                                    )
    

    

    # Add the BD derived values: name, Teff, logg, met, distance_pc, radius_Rjup
    if object_name == 'Ross458C':
        bd_object.bd_info('Ross458C','804','4.09','0.23', 11.509, 0.68 )
    if object_name == 'HD3651B':
        bd_object.bd_info('HD3651B','818','3.94','-0.22', 11.134, 0.81 )
    if object_name == 'GJ570D':
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

    
    XminXmax_Stand = train_cnn_regression_class.standardize_X_ColumnWise.transform(df_MinMax_obs.values)

    # X_Norm = (bd_object.df_flux_object.values[0] - bd_object.df_flux_object.min(axis=1)[0]) / (bd_object.df_flux_object.max(axis=1)[0] - bd_object.df_flux_object.min(axis=1)[0])
    # X_Norm = X_Norm * (1. - 0.) + 0.
    
    bd_mean = bd_object.df_flux_object.mean(axis=1)[0]  
    bd_std = bd_object.df_flux_object.std(axis=1)[0]     

    X_Scaled = (bd_object.df_flux_object.values[0] - bd_mean) / bd_std
    

    
    y_pred_train = np.array(train_cnn_regression_class.trained_model.predict([X_Scaled[::-1].reshape(1,104),XminXmax_Stand], verbose=0))[:,:,0].T
    y_pred_train_ = train_cnn_regression_class.standardize_y_ColumnWise.inverse_transform( y_pred_train )
    y_pred_train_ [:,3] = 10**y_pred_train_[:,3]
    y_pred = y_pred_train_
    
    # print(y_pred)
    
    # ********************************* 
    
    targets = dict ( zip( ['gravity', 'c_o_ratio', 'metallicity', 'temperature'] , [y_pred[0][i] for i in range(4)]) )
    
    if print_results_:
        print_results_fun(targets, print_title= 'Predicted Targets:')
    
    if plot_predicted_vs_observed_: 
        plot_predicted_vs_observed(training_datasets = dataset, 
                                   wl = wl,
                                    predicted_targets_dic = targets,
                                    object_name = object_name,
                                    bd_object_class = bd_object,
                                    print_results=True, 
                                  )

    # plot_predicted_vs_observed(training_datasets = dataset, 
    #                                     predicted_targets_dic = targets,
    #                                     object_name = object_name )    
    # ********************************* 

        
        
        
    if predict_from_random_spectra:
        for i in range(random_spectra_num):
            # Comment from Natasha: 
            spectra = pd.DataFrame( np.random.normal(obs_data['F_lambda'] , obs_data['F_lambda_error'] ) ,
                                     columns=['F_lambda'])
            # spectra = pd.DataFrame( [random.uniform(fluxcal.F_lambda[i]-1*fluxcal.F_lambda_error[i], 
                                 # fluxcal.F_lambda[i]+1*fluxcal.F_lambda_error[i]) 
                                 # for i in range(len(fluxcal))],columns=['F_lambda'] )
            
            


            # Process the dataset
            spectra['F_lambda'] = spectra['F_lambda'].mask(spectra['F_lambda'].lt(0),0)
            spectra['F_lambda'].replace(0, np.nan, inplace=True)
            spectra['F_lambda'].interpolate(inplace=True)


            # Instintiate Process Observational Dataset class
            bd_object_generated = ProcessObservationalDataset( feature_values_obs = spectra['F_lambda'].values,
                                                              feature_values_obs_err = obs_data['F_lambda_error'].values,
                                             feature_names_obs  = obs_data['wl'].values, 
                                             feature_names_synthetic = wl['wl'].values,
                                            )
    # # Instintiate ProcessObservationalDataset class
    # bd_object = ProcessObservationalDataset( feature_values_obs = fluxcal['F_lambda'].values,
    #                                         feature_values_obs_err = fluxcal['F_lambda_error'].values,
    #                                          feature_names_obs  = fluxcal['wl'].values, 
    #                                          feature_names_synthetic = wl['wl'].values,
    #                                          # feature_values_obs_error = fluxcal['F_lambda_error'].values,
    #                                 )

            # add the BD derived values: name, Teff, logg, met, distance_pc, radius_Rjup
            if object_name == 'Ross458C':
                bd_object_generated.bd_info('Ross458C','804','4.09','0.23', 11.509, 0.68 )
            if object_name == 'HD3651B':
                bd_object_generated.bd_info('HD3651B','818','3.94','-0.22', 11.134, 0.81 )
            if object_name == 'GJ570D':
                bd_object_generated.bd_info('GJ570D','818','3.94','-0.22', 5.884, 0.79 )    
    



            bd_object_generated.flux_interpolated(print_results=False, 
                                        plot_results=False,
                                        use_spectres=True
                                       )
            bd_object_generated.df_flux_object    
    
            # bd_object_generated.flux_interpolated(print_results=False, plot_results=False)
            # bd_object_generated.df_flux_object.values
            spectra_list.append(bd_object_generated.df_flux_object.values)
            
            # ********************************* 
            bd_object_generated.df_flux_object_min = bd_object_generated.df_flux_object.min(axis=1)
            bd_object_generated.df_flux_object_max = bd_object_generated.df_flux_object.max(axis=1)

            df_MinMax_obs = pd.DataFrame(( bd_object_generated.df_flux_object_min, bd_object_generated.df_flux_object_max)).T

            # xxx2 = train_cnn_regression.standardize_X_ColumnWise.transform(df_MinMax_obs.values)
            XminXmax_Stand = train_cnn_regression_class.standardize_X_ColumnWise.transform(df_MinMax_obs.values)

            
            # X_std = (bd_object_generated.df_flux_object.values[0] - bd_object_generated.df_flux_object.min(axis=1)[0]) / (bd_object_generated.df_flux_object.max(axis=1)[0] - bd_object.df_flux_object.min(axis=1)[0])
            # X_scaled = X_std * (1. - 0.) + 0.

            bd_mean = bd_object_generated.df_flux_object.mean(axis=1)[0]  
            bd_std = bd_object_generated.df_flux_object.std(axis=1)[0]     

            X_Scaled = (bd_object_generated.df_flux_object.values[0] - bd_mean) / bd_std
    
            # bd_mean = bd_object_generated.df_flux_object.mean(axis=1)[0]  
            # bd_std = bd_object_generated.df_flux_object.std(axis=1)[0]     

            # X_scaled = (bd_object_generated.df_flux_object.values[0] - bd_mean) / bd_std

            y_pred_train = np.array(train_cnn_regression_class.trained_model.predict([X_Scaled[::-1].reshape(1,104),XminXmax_Stand], verbose=0) )[:,:,0].T
            y_pred_train_ = train_cnn_regression_class.standardize_y_ColumnWise.inverse_transform( y_pred_train )
            y_pred_train_ [:,3] = 10**y_pred_train_[:,3]
            y_pred = y_pred_train_
            # ********************************* 
    
    
            param_list.append(  y_pred[0] )

        df_random_pred = pd.DataFrame(param_list, columns=['logg' ,'c_o' ,'met' ,'T'] )
        display(df_random_pred.describe())
            


        if plot_randomly_generated_spectra:
            p = figure(title=object_name+": Randomly generated spectra within 1σ", 
                       x_axis_label='Features (Wavelength [𝜇m])', y_axis_label='Flux (Fν)',
                       width=1000, height=300,
                       y_axis_type="log", background_fill_color="#fafafa"
                      )
            

            for i in range(0,random_spectra_num,int(random_spectra_num/5)):
                    p.line(wl.wl.values[::-1],spectra_list[i][0], 
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


        p.legend.location = "top_right"
        p.legend.background_fill_color = 'white'
        p.legend.background_fill_alpha = 0.5
    
        show(p)

            
        if print_df_describe:
            display(df_random_pred.describe())
            
            
        if plot_histogram:
            plt.figure()
            df_random_pred.hist()
            plt.show()
        
        if plot_boxplot_hist: 
            boxplot_hist(df_random_pred['logg'],  x_label=r'$\log g$', xy_loc=[0.05,0.98],)
            boxplot_hist(df_random_pred['T'],     x_label=r'$T_{eff}$', xy_loc=[0.05,0.98],)
            boxplot_hist(df_random_pred['c_o'],   x_label=r'C/O', xy_loc=[0.05,0.98],)
            boxplot_hist(df_random_pred['met'],   x_label=r'[M/H]', xy_loc=[0.05,0.98],)
            

    if plot_predicted_vs_observed_: 
        targets = dict ( zip( ['gravity', 'c_o_ratio', 'metallicity', 'temperature'] ,  list( df_random_pred.agg(np.mean) ) ) )
        plot_predicted_vs_observed(training_datasets = dataset, 
                                    wl = wl,
                                    predicted_targets_dic = targets,
                                    object_name = object_name,
                                    bd_object_class = bd_object,
                                    print_results=True, 
                                  )

    if plot_predicted_vs_spectra_errorbar_: 
        targets = dict ( zip( ['gravity', 'c_o_ratio', 'metallicity', 'temperature'] ,  list( df_random_pred.agg(np.mean) ) ) )
        plot_predicted_vs_spectra_errorbar(
                              object_name = object_name, 
                              features = obs_data['wl'], 
                              feature_values = bd_object.Fnu_obs_absolute,
                              error = np.array(bd_object.Fnu_obs_absolute_err),
                          training_datasets = dataset,  
                           wl = wl,
                           predicted_targets_dic = targets,
                           bd_object_class = bd_object,
                           print_results_ = True)
        
    # if plot_predicted_intervals_vs_spectra_errorbar_:
        
            
    return df_random_pred, spectra_list

        
"""    
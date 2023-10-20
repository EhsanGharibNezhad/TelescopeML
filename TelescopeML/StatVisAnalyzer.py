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

import pprint

# ******* Data Visulaization Libraries ****************************
# from bokeh.palettes import colorblind
import seaborn as sns
import matplotlib.pyplot as plt


from bokeh.plotting import output_notebook

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource


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



def regression_report(trained_model,
                      Xtrain, Xtest, ytrain, ytest,
                      target_i,
                      xy_top=None, xy_bottom=None, __print_results__= False):
    """
    Generate a regression report for the trained ML/CNN model.

    Parameters
    -----------
    trained_model : object
        Trained regression model.
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
    y_pred_train = np.array(trained_model.trained_model.predict(Xtrain))[:, :, 0].T
    y_pred_train_list = trained_model.standardize_y_ColumnWise.inverse_transform(y_pred_train)
    y_pred_train_list[:, 3] = 10 ** y_pred_train_list[:, 3]

    y_act_train_list = trained_model.standardize_y_ColumnWise.inverse_transform(ytrain)
    y_act_train_list[:, 3] = 10 ** y_act_train_list[:, 3]

    # Apply the trained ML model on the test set to predict the targets
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

        if __print_results__:
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

        # Plot scatter figures of predicted vs actual values
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
        target_name = ['Gravity', 'C_O_ratio', 'Metallicity', 'Temperature'][i]
        plt.savefig(f'../outputs/figures/regression_report_{target_name}.pdf', format='pdf')
        plt.show()

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
        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented']).columns)

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
                        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented'])

                    y = df_to_interpolate_['c_o_ratio'].to_numpy()

                    column_grid = data.columns.to_numpy().astype(float)
                    values = df_to_interpolate_.drop(
                        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented']).to_numpy()

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
                    df_interpolated_['is_augmented'] = 'no'

                    df_interpolated_all = pd.concat([df_interpolated_, df_interpolated_all], ignore_index=True)

    # ***************************************************************************************

    my_list_g = list(df_interpolated_all['gravity'].sort_values().unique())
    my_list_met = list(df_interpolated_all['metallicity'].sort_values().unique())
    my_list_c_o = list(df_interpolated_all['c_o_ratio'].sort_values().unique())
    my_list_T = list(df_interpolated_all['temperature'].sort_values().unique())

    df_interpolated_all2 = df_interpolated_all
    df_interpolated_all2.drop_duplicates(inplace=True)

    df_interpolated_ = pd.DataFrame(columns=df_interpolated_all2.drop(
        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented']).columns)

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
                        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented'])

                    y = df_to_interpolate_['metallicity'].to_numpy()

                    column_grid = data.columns.to_numpy().astype(float)
                    values = df_to_interpolate_.drop(
                        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented']).to_numpy()

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
                    df_interpolated_['is_augmented'] = 'no'

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
        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented']).columns)

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
                        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented'])

                    y = df_to_interpolate_['temperature'].to_numpy()
                    # print(y)

                    column_grid = data.columns.to_numpy().astype(float)
                    values = df_to_interpolate_.drop(
                        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented']).to_numpy()

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
                    df_interpolated_['is_augmented'] = 'no'

                    df_interpolated_all = pd.concat([df_interpolated_, df_interpolated_all], ignore_index=True)

    # ******************************************************************************************

    my_list_g = list(df_interpolated_all['gravity'].sort_values().unique())
    my_list_met = list(df_interpolated_all['metallicity'].sort_values().unique())
    my_list_c_o = list(df_interpolated_all['c_o_ratio'].sort_values().unique())
    my_list_T = list(df_interpolated_all['temperature'].sort_values().unique())

    df_interpolated_all2 = df_interpolated_all
    df_interpolated_all2.drop_duplicates(inplace=True)

    df_interpolated_ = pd.DataFrame(columns=df_interpolated_all2.drop(
        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented']).columns)

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
                        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented'])

                    y = df_to_interpolate_['gravity'].to_numpy()

                    column_grid = data.columns.to_numpy().astype(float)
                    values = df_to_interpolate_.drop(
                        columns=['gravity', 'temperature', 'c_o_ratio', 'metallicity', 'is_augmented']).to_numpy()

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
                    df_interpolated_['is_augmented'] = 'no'

                    df_interpolated_all = pd.concat([df_interpolated_, df_interpolated_all], ignore_index=True)


    df_interpolated_all.drop_duplicates(inplace=True)

    df_interpolated_final = df_interpolated_all[
        (df_interpolated_all['temperature'] == predicted_targets_dic['temperature']) &
        (df_interpolated_all['c_o_ratio'] == predicted_targets_dic['c_o_ratio']) &
        (df_interpolated_all['metallicity'] == predicted_targets_dic['metallicity']) &
        (df_interpolated_all['gravity'] == predicted_targets_dic['gravity']) &
        (df_interpolated_all['is_augmented'] == 'no')]

    return df_interpolated_final


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
    if x_label == 'C/O':
        x_label = 'c_o_ratio'
    if x_label == '[M/H]':
        x_label = 'metallicity'
    plt.savefig(f'../outputs/figures/boxplot_hist_{x_label}.pdf', format='pdf')

    plt.show()


# def plot_predicted_vs_observed(training_datasets,
#                                wl,
#                                predicted_targets_dic,
#                                object_name,
#                                df_Fnu_obs_absolute_intd,
#                                __print_results__ = False,
#                               ):
#
#
#
#     ypred = list( predicted_targets_dic.values() )
#
#     filtered_df = interpolate_df(dataset=training_datasets,
#                        predicted_targets_dic = predicted_targets_dic,
#                        print_results_ = False)
#
#     if __print_results__:
#         print('*'*30+'Filtered and Interpolated training data based on the ML predicted parameters'+'*'*30)
#         print(filtered_df)
#
#
#     p = figure(
#         # title=f'{object_name} [XStand, yStand] Predicted: '+', '.join([['logg= ','C/O= ', 'Met= ', 'T= '][i]+str(np.round(y_pred[0][i],2)) for i in  range(4)]),
#                x_axis_label='Features (Wavelength [𝜇m])',
#                y_axis_label='Flux (F𝜈)',
#                width=800, height=300,
#                y_axis_type = 'log')
#
#     # Add the scatter plot
#
#     p.line(x =wl['wl'] , y=filtered_df.drop(columns=['gravity', 'c_o_ratio', 'metallicity', 'temperature','is_augmented']).values[0],
#            line_width = 1,
#            legend_label= 'ML Predicted:'+', '.join([['log𝑔= ','C/O= ', '[M/H]= ', 'T= '][i]+str(np.round(ypred[i],2)) for i in  range(4)]))
#
#     if __print_results__:
#         print(df_Fnu_obs_absolute_intd.iloc[:, ::-1])
#
#     p.line(x = wl['wl'] , y = df_Fnu_obs_absolute_intd.iloc[:, ::-1].values[0],
#            line_color = 'orange', line_width = 2,
#            legend_label='Observational')
#
#     p.circle(x = wl['wl'] , y = df_Fnu_obs_absolute_intd.iloc[:, ::-1].values[0],#.iloc[:,4:-1].values[0],
#            line_width = 2,
#            color='orange'
#             )
#
#     # Increase size of x and y ticks
#     p.title.text_font_size = '12pt'
#     p.xaxis.major_label_text_font_size = '12pt'
#     p.xaxis.axis_label_text_font_size = '12pt'
#     p.yaxis.major_label_text_font_size = '12pt'
#     p.yaxis.axis_label_text_font_size = '12pt'
#
#
#     p.legend.location = "top_right"
#     p.legend.background_fill_color = 'white'
#     p.legend.background_fill_alpha = 0.5
#
#
#     show(p)


def plot_spectra_errorbar(object_name,
                          x_obs,
                          y_obs,
                          y_obs_err,
                          y_label = "Flux (F𝜈) [erg/s/cm2/Hz]",
                          title_label = None,
                          data_type='x_y_yerr'):
    # Create the figure
    p = figure(title=f"{object_name}: Calibrated Observational Spectra" if title_label is None else title_label,
               x_axis_label="Features (Wavelength [𝜇m])",
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
    p = figure(title=f"{object_name}: Calibrated Observational VS. Predicted Spectra",
               x_axis_label="Features (Wavelength [𝜇m])",
               y_axis_label="Flux (F𝜈) [erg/s/cm2/Hz]",
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
        y=filtered_df.drop(columns=['gravity', 'c_o_ratio', 'metallicity', 'temperature', 'is_augmented']).values[0],
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

def plot_pred_vs_obs_errorbar_stat(  stat_df,
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
        x_axis_label='Features (Wavelength [μm])',
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
            x_axis_label='Features (Wavelength [μm])',
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
        x_axis_label='Features (Wavelength [𝜇m])',
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

# Example usage with df = 103 and chi_square_list containing chi-square statistics
# df_value = 103
# chi_square_list = [93, 32,  150.456789123, 120.789123456]  # Replace with actual chi-square statistics



# check_chi_square(103, chi_square_list)


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
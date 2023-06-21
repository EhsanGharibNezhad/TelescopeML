# Import functions from other modules ============================
from io_funs import LoadSave
# from train_ml_regression_2 import TrainMlRegression

# Import python libraries ========================================

# Dataset manipulation libraries
import pandas as pd
import numpy as np
import pickle as pk
import os
from scipy import stats

# ML algorithm libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Optimization libraries
from scipy import stats
import skopt
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective
from skopt.plots import plot_evaluations


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

# ===============================================================================
# ==================                                           ==================
# ==================            check_trained_model_results    ==================
# ==================                                           ==================
# ===============================================================================        

class CheckTrainedMlRegression:
    """
    perform traditional ML training

    Tasks
    ------
    - Process dataset: Scale, split_train_val_test
    - Train ML model
    - Optimize
    - Feature reduction: PCA, RFE
    - Output: Save trained models, metrics

    Attributes
    -----------
    - feature_values: array
            flux arrays in XXXX [write the dimension]
    - feature_names: list
            name of wavelength in micron
    - target_values: array
            target variable array: Temperature, Gravity, Carbon_to_Oxygen, Metallicity
    - is_tuned: str
            To optimize the hyperparameters: 'yes' or 'no'
    - param_grid: dic
            ML hyperparameters to be tuned.
            Note: param_grid is used if is_tuned = 'yes'
    - spectral_resolution: int
            resolution of the synthetic spectra used to generate the dataset
    - is_feature_improved: str
        options:
            'no': all features
            'pca': used Principal component analysis method for dimensionality reduction
            'RFE': used Recursive Feature Elimination (RFE) for Feature Selection
    - is_augmented: str
        options:
            'no': used native dataset
            [METHOD]: augmented dataset like adding noise etc.
    - ml_model_str: str
            name of the ML model

    Outputs
    --------
    - Trained ML models

    """

    def __init__(self,
                 trained_model,
                 feature_values,
                 feature_names,
                 target_values,
                 target_name,
                 is_tuned,
                 param_grid,
                 spectral_resolution,
                 is_feature_improved,
                 is_augmented,
                 ml_model_str,
                 ):

        self.feature_values = feature_values
        self.feature_names = feature_names
        self.target_values = target_values
        self.target_name = target_name
        self.is_tuned = is_tuned
        self.param_grid = param_grid
        self.spectral_resolution = spectral_resolution
        self.feature_names = feature_names
        self.is_feature_improved = is_feature_improved
        self.is_augmented = is_augmented
        self.ml_model_str = ml_model_str

        if trained_model == None:
            trained_model = LoadSave(self.ml_model_str,
                     self.is_feature_improved,
                     self.is_augmented,
                     self.is_tuned,).load_or_dump_trained_object(trained_object = None,
                                                                indicator = 'TrainedModel',
                                                                load_or_dump = 'load')
        self.trained_model = trained_model



    def split_train_test(self,
                         test_size = 0.1):
        """
        split the loaded set into train and test sets

        Inputs
        ------
            - self.feature_values
            - self.target_values
            - test_size: default is 10% kept for testing the final trained model
            - shuffle: default is 'True'
            - random_state: default is 42
        Return
        -------
            - self.X_train, self.X_test: Used to train the machine learning model and evaluate it later.
            - self.y_train, self.y_test: Targets used for train/test the models

        Refs
        -----
            - SciKit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.feature_values,
                                                                                self.target_values,
                                                                                test_size = test_size,
                                                                                shuffle = True,
                                                                                random_state = 42,
                                                                            )
        

    def StandardScaler_X (self, 
                          indicator = 'StandardScaler_X'):
        """
        Utilize the trained StandardScaler and apply them on  X_train and X_test
        """
        StandardScaler_X = LoadSave( self.ml_model_str,
                                     self.is_feature_improved,
                                     self.is_augmented,
                                     self.is_tuned,).load_or_dump_trained_object(trained_object = None,
                                                                            indicator = indicator,
                                                                            load_or_dump = 'load')

        self.StandardScaler_X = StandardScaler_X

        if indicator == 'StandardScaler_row_X':
            SS = StandardScaler()
            self.X_train = SS.fit_transform(self.X_train.T).T
            self.X_test  = SS.fit_transform(self.X_test.T).T

        elif indicator == 'StandardScaler_col_X':
            self.X_train = StandardScaler_X.transform(self.X_train)
            self.X_test  = StandardScaler_X.transform(self.X_test)          
        

    def StandardScaler_y (self, 
                          indicator = 'StandardScaler_y'):
        """
        Utilize the trained StandardScaler and apply them on  y_train and y_test
        """        
        StandardScaler_y = LoadSave(self.ml_model_str,
                                 self.is_feature_improved,
                                 self.is_augmented,
                                 self.is_tuned,).load_or_dump_trained_object(trained_object = None,
                                                                            indicator = indicator,
                                                                            load_or_dump = 'load')
        
            
        self.y_train = StandardScaler_y.transform(self.y_train)
        self.y_test  = StandardScaler_y.transform(self.y_test )
        self.StandardScaler_y = StandardScaler_y
        

        
    def StandardScaler_row_X_3(self,
                             print_model = False):
        """
        Standardize observations/instances/row variables by removing the mean and scaling to unit variance.
        Transform your data such that its distribution will have a mean value 0 and standard deviation of 1
        The standard score of a sample x is calculated as: z = (x - u) / s
        ... u: mean of the dataset, s: standard deviation

        Inputs
        ------
            - Build-in "StandardScaler()" class
            - Attributes: self.X_train and self.X_test sets
        Outputs
        -------
            - Scaled self.X_train and self.X_test sets
            - Trained scaled/fitted "StandardScaler()"

        Ref: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        """
        ss = StandardScaler()
        self.X_train = ss.fit_transform(self.X_train.T).T
        self.X_val= ss.fit_transform(self.X_val.T).T
        self.X_test = ss.fit_transform(self.X_test.T).T
        self.StandardScaler_row_X = ss


        LoadSave(self.ml_model_str,
                     self.is_feature_improved,
                     self.is_augmented,
                     self.is_tuned,).load_or_dump_trained_object(trained_object = self.StandardScaler_row_X,
                                                                indicator = 'StandardScaler_row_X',
                                                                load_or_dump = 'dump')

        if print_model == True:
            print(trained_StandardScaler)
        
        
    def StandardScaler_col_X_3(self):
        """
        Standardize observations/instances/row variables by removing the mean and scaling to unit variance.
        Transform your data such that its distribution will have a mean value 0 and standard deviation of 1
        The standard score of a sample x is calculated as: z = (x - u) / s
        ... u: mean of the dataset, s: standard deviation

        Inputs
        ------
            - Build-in "StandardScaler()" class
            - Attribitutes: self.X_train and self.X_test sets
        Outputs
        -------
            - Scaled self.X_train, self.X_val, and self.X_test sets
            - Trained scaled/fitted "StandardScaler()"

        Ref: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        """

        ss = StandardScaler()

        self.X_train = ss.fit_transform(self.X_train)
        self.X_val = ss.transform(self.X_val)
        self.X_test = ss.transform(self.X_test)
        self.StandardScaler_col_X_3 = ss

        LoadSave(self.ml_model_str,
                     self.is_feature_improved,
                     self.is_augmented,
                     self.is_tuned,
                                ).load_or_dump_trained_object(trained_object=ss,
                                                 indicator='StandardScaler_col_X_3',
                                                 load_or_dump='dump')
        
        
        
        

    def StandardScaler_column_y_3(self):
        """
        Standardize observations/instances/row variables by removing the mean and scaling to unit variance.
        Transform your data such that its distribution will have a mean value 0 and standard deviation of 1
        The standard score of a sample x is calculated as: z = (x - u) / s
        ... u: mean of the dataset, s: standard deviation

        Inputs
        ------
            - Build-in "StandardScaler()" class
            - Attribitutes: self.X_train and self.X_test sets
        Outputs
        -------
            - Scaled self.X_train and self.X_test sets
            - Trained scaled/fitted "StandardScaler()"

        Ref: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        """

        ss = StandardScaler()
        self.y_train = ss.fit_transform(self.y_train)
        self.y_val = ss.transform(self.y_val)
        self.y_test = ss.transform(self.y_test)
        self.StandardScaler_column_y = ss


        LoadSave(self.ml_model_str,
                 self.is_feature_improved,
                 self.is_augmented,
                 self.is_tuned,
                 ).load_or_dump_trained_object(trained_object = ss,
                                              indicator = 'StandardScaler_col_y',
                                              load_or_dump = 'dump')        
                
        
                

    def split_train_validation_test(self,
                                    test_size = 0.1,
                                    val_size = 0.1):
        """
        split the loaded set into train, validation and test sets

        Inputs
        ------
            - self.feature_values
            - self.target_values
            - test_size: default is 10% kept for testing the final trained model
            - shuffle: default is 'True'
            - random_state: default is 42
        Return
        -------
            - self.X_train, self.X_test: Used to train the machine learning model and evaluate it later.
            - self.y_train, self.y_test: Targets used for train/test the models

        Refs
        -----
            - SciKit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.feature_values,
            self.target_values,
            test_size = 0.10,
            shuffle = True,
            random_state = 42,)
        self.X_test, self.y_test = X_test, y_test
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train,
            y_train,
            test_size = val_size,
            shuffle = True,
            random_state = 42,)        
        
    def check_BayesSearchCV(self,
                            print_results = True,
                            plot_results = True,
                            print_cv_results_df = False, 
                           ):
        if print_results:
            print(' ==============    Optimal HyperParameters    ============== ')
            # display(self.optimized_params)
            print("total_iterations", self.trained_model.total_iterations)
            print("val. score: %s" % self.trained_model.best_score_)
            print("test score: %s" % self.trained_model.score(self.X_test, self.y_test))
            print("best params: %s" % str(self.trained_model.best_params_))

        if plot_results:
            plot_evaluations(self.trained_model.optimizer_results_[0],
                             # bins=10,
                             dimensions=[xx.replace('estimator__', '') for xx in list(self.param_grid.keys())]
                             )
            
        if print_cv_results_df:
            display ( pd.DataFrame(self.trained_model.cv_results_) )
            
            

    def check_feature_selection_RFE (self,
                               trained_model = None,
                               is_tuned = None,
                               num_features2drop = None,
                               print_ = True,
                               plot_ = True,
                              ):
        """
        RFE: recursive Feature Elimination
        """
        if is_tuned == None:
            is_tuned = self.is_tuned
            
        if is_tuned == 'yes':
            feature_importance = trained_model.best_estimator_.feature_importances_
        elif is_tuned == 'no':
            feature_importance = trained_model.feature_importances_
            
        if trained_model == None:
            trained_model = self.trained_model

        feature_importnace_df = pd.DataFrame (  sorted(list(zip(self.feature_names,feature_importance))), 
                       columns=['feature_names',  'feature_importnace']).sort_values(by='feature_importnace',
                                                   ignore_index = True)
        self.feature_importnace_df = feature_importnace_df
        
        if num_features2drop != None:
            self.features2keep_df = feature_importnace_df['feature_names'][num_features2drop:]
            print('-------',self.features2keep_df)
            self.load_or_dump_features2keep_df(load_or_dump='dump')    
            

            
            
        if print_:
            print('feature number -- feature_importnace -- feature_importnace%')
            print(self.feature_importnace_df.head(10))
            
        if plot_:
            
            # Plot feature labels (sorted) vs. their importnace
            feature_importnace_df = pd.DataFrame (  sorted(list(zip(self.feature_names,feature_importance))), 
                           columns=['feature_names',  'feature_importnace']).sort_values(by='feature_importnace',
                                                       ignore_index = True)
            feature_importnace_df = feature_importnace_df

            plt.figure(figsize=(25,5))
            plt.bar(feature_importnace_df['feature_names'].values,
                feature_importnace_df['feature_importnace']/feature_importnace_df['feature_importnace'].max())
            plt.xticks(rotation=90,fontsize=12)
            plt.show()


            # Plot feature labels (UN-sorted) vs. their importnace
            feature_importnace_df = pd.DataFrame (  list(zip(self.feature_names,feature_importance)), 
               columns=['feature_names',  'feature_importnace'])

            plt.figure(figsize=(25,5))
            plt.bar(feature_importnace_df['feature_names'].values,
                feature_importnace_df['feature_importnace']/feature_importnace_df['feature_importnace'].max())
            plt.xticks(rotation=90,fontsize=12)
            plt.show()

            
    def check_feature_reduction_PCA (self,
                                    trained_model_pca,
                                    is_tuned,
                                    plot_print_ = True,
                                   ):
        if trained_model_pca == None:
            pca_filename = os.path.join(
                                  '../outputs/trained_models/',
                                  'PCA_' +
                                   self.ml_model_str+'_'+ 
                                   self.target_name+
                                  '_feature_improved_PCA'+'.'+ 
                                  self.is_tuned+ '_tuned')
                
            trained_model_pca = pk.load(open(pca_filename, 'rb'))
    
    
        if plot_print_:
            plt.figure()
            #             sns.set(style='whitegrid')
            plt.plot(np.cumsum(trained_model_pca.explained_variance_ratio_))
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')
            plt.axvline(linewidth=4, color='r', linestyle = '--', x=10, ymin=0, ymax=1)
            print(plt.show())
            evr = trained_model_pca.explained_variance_ratio_
            cvr = np.cumsum(trained_model_pca.explained_variance_ratio_)
            pca_df = pd.DataFrame()
            pca_df['Cumulative Variance Ratio'] = cvr
            pca_df['Explained Variance Ratio'] = evr
            display(pca_df.head(10))
            
        
            pca_dims = []
            for x in range(0, len(pca_df)):
                pca_dims.append('PCA Component {}'.format(x))
            pca_test_df = pd.DataFrame(trained_model_pca.components_, index=pca_dims)
            display(pca_test_df.head(10).T)
            plt.show()
            

            
    from scipy import stats

    def regression_report_CNN(trained_model, 
                              Xtrain, Xtest, ytrain, ytest,
                              yScaler,
                              target_i,
                              xy_top = [0.55, 0.85],
                              xy_bottom = [0.05, 0.8],
                              print_results = True):

        y_pred_train =  np.array ( trained_model.predict(Xtrain) )[:,:,0,0].T
        residual_train_list = yScaler.inverse_transform( np.array(y_pred_train)) - yScaler.inverse_transform( ytrain)
        # print(residual_train_list, residual_train_list[:,-1])
        y_pred_test =  np.array ( trained_model.predict(Xtest) )[:,:,0,0].T
        residual_test_list = yScaler.inverse_transform( np.array(y_pred_test)) - yScaler.inverse_transform( ytest)

        f, axs = plt.subplots(2,1,
                            figsize=(5,5),
                            sharey=False,
                            sharex=True,
                            gridspec_kw=dict(height_ratios=[1,3])
                            )

        # y_pred_train =  np.array ( trained_cnn_model['model'].predict( Xtrain ) )[:,:,0,0].T
        residual_train = residual_train_list[:,target_i] 
        residual_test  = residual_test_list[:,target_i]  


        slope_test, intercept_test, r_value_test, p_value_test, std_err_test = stats.linregress(y_pred_train[:,target_i] , ytrain[:,target_i])
        slope_train, intercept_train, r_value_train, p_value_train, std_err_train = stats.linregress(y_pred_test[:,target_i] , ytest[:,target_i])


        mean_test = np.round(np.mean(residual_test),2)
        std_test = np.round(np.std(residual_test),2)
        mean_train = np.round(np.mean(residual_train),2)
        std_train = np.round(np.std(residual_train),2)



        skew_test = stats.skew(residual_test)
        skew_train = stats.skew(residual_train)

        if print:
            print('\n\n----------------------- Test ------------------------')
            print(' slope, intercept, r_value, p_value, std_err')
            print(slope_test, intercept_test, r_value_test, p_value_test, std_err_test)
            print('------- mean, std -----------')
            print( mean_test, std_test)
            print('------- Skewness -----------')
            print(skew_test)

            print('\n----------------------- Train ------------------------')
            print(' slope, intercept, r_value, p_value, std_err')
            print(slope_train, intercept_train, r_value_train, p_value_train, std_err_train )
            print('------ mean, std -----------')
            print( mean_train, std_train)
            print('------- Skewness -----------')
            print(skew_train)
            print('------------------\n\n\n')

        sns.histplot(data= residual_train, 
                    ax=axs[0],
                    label='train', 
                    alpha = 0.7, bins=19, log_scale=False, stat='percent', legend=True, linewidth=0
                    )

        sns.histplot(data= residual_test, 
                    label='test', 
                    ax=axs[0],
                    alpha = 0.3, bins=19, stat='percent', legend=True,linewidth=0
                )

        axs[0].set_ylim((1e-1,100))
        axs[0].set_yscale('log')


        sns.scatterplot(
                    y=pd.DataFrame(ytrain[:,target_i])[0],
                    x= pd.DataFrame(residual_train)[0],
                label='train', 
                                ax=axs[1],

                    alpha = 0.7, legend=False,
                    )

        sns.scatterplot(
                    y=pd.DataFrame(ytest[:,target_i])[0],
                    x= pd.DataFrame(residual_test)[0],
                label='test', 
                                ax=axs[1],

                    alpha = 0.7, legend=False,
                    )

        axs[0].set_ylim((1e-1,100))
        axs[0].set_yscale('log')


        axs[1].set_xlabel('Residual value', fontsize = 12)
        axs[1].set_ylabel('Actual value', fontsize = 12)
        axs[0].set_ylabel('Probability %', fontsize = 12)


        axs[0].annotate(r'$\tilde{\mu}_{{\rm 3, train}}$= '+f'{np.round(skew_train,2)}', 
                            fontsize=11, xy=(xy_top[0],xy_top[1]+0.08), xycoords='axes fraction')
        axs[0].annotate(r'$\tilde{\mu}_{{\rm 3, test}}$ = '+f'{np.round(skew_test,2)}', 
                            fontsize=11, xy=(xy_top[0],xy_top[1]-0.08), xycoords='axes fraction')

        axs[1].annotate(r'R$^2_{\rm train}$='+f'{"%0.2f"%r_value_train**2} [{"%0.2f"%abs(mean_train)}$\pm${"%0.2f"%std_train}]', 
                            fontsize=11, xy=(xy_bottom[0],xy_bottom[1]+0.06), xycoords='axes fraction')
        axs[1].annotate(r'R$^2_{\rm test}$ ='+f'{np.round(r_value_test**2,2)} [{"%0.2f"%mean_test}$\pm${"%0.2f"%std_test}]',
                            fontsize=11, xy=(xy_bottom[0],xy_bottom[1]-0.06), xycoords='axes fraction')





        axs[1].legend(loc='lower left', fontsize = 11)
        # plt.yscale('log')

        f.tight_layout()
        plt.show()       

            
    def regression_report(self,
                          target_i,
                          trained_model = None,
                          xy_top = [0.55, 0.85],
                          xy_bottom = [0.05, 0.8],
                          print_results=False):

        f, axs = plt.subplots(2, 1,
                              figsize = (5, 5),
                              sharey = False,
                              sharex = True,
                              gridspec_kw = dict(height_ratios=[1, 3])
                              )

        if trained_model == None:
            trained_model = self.trained_model

        residual_train = trained_model.predict(self.X_train)[:, target_i] - self.y_train[:, target_i]
        residual_test = trained_model.predict(self.X_test)[:, target_i] - self.y_test[:, target_i]

        slope_test, intercept_test, r_value_test, p_value_test, std_err_test = stats.linregress(
            trained_model.predict(self.X_train)[:, target_i],
            self.y_train[:, target_i])
        slope_train, intercept_train, r_value_train, p_value_train, std_err_train = stats.linregress(
            trained_model.predict(self.X_test)[:, target_i],
            self.y_test[:, target_i])

        mean_test = np.round(np.mean(residual_test), 2)
        std_test = np.round(np.std(residual_test), 2)
        mean_train = np.round(np.mean(residual_train), 2)
        std_train = np.round(np.std(residual_train), 2)

        skew_test = stats.skew(residual_test)
        skew_train = stats.skew(residual_train)

        if print_results:
            print('----------------------- Test ------------------------')
            print(' slope, intercept, r_value, p_value, std_err')
            print(slope_test, intercept_test, r_value_test, p_value_test, std_err_test)
            print('------- mean, std -----------')
            print(mean_test, std_test)
            print('------- Skewness -----------')
            print(skew_test)

            print('\n----------------------- Train ------------------------')
            print(' slope, intercept, r_value, p_value, std_err')
            print(slope_train, intercept_train, r_value_train, p_value_train, std_err_train)
            print('------- mean, std -----------')
            print(mean_train, std_train)
            print('------- Skewness -----------')
            print(skew_train)
            print('------------------\n')

        sns.histplot(data=residual_train,
                     ax=axs[0],
                     label='train',
                     alpha=0.7, bins=19, log_scale=False, stat='percent', legend=True, linewidth=0
                     )

        sns.histplot(data=residual_test,
                     label='test',
                     ax=axs[0],
                     alpha=0.3, bins=19, stat='percent', legend=True, linewidth=0
                     )

        axs[0].set_ylim((1e-1, 100))
        axs[0].set_yscale('log')

        sns.scatterplot(
            y=pd.DataFrame(self.y_train[:, target_i])[0],
            x=pd.DataFrame(residual_train)[0],
            label='train',
            ax=axs[1],

            alpha=0.7, legend=False,
        )

        sns.scatterplot(
            y=pd.DataFrame(self.y_test[:, target_i])[0],
            x=pd.DataFrame(residual_test)[0],
            label='test',
            ax=axs[1],

            alpha=0.7, legend=False,
        )

        axs[0].set_ylim((1e-1, 100))
        axs[0].set_yscale('log')

        axs[1].set_xlabel('Residual value', fontsize=12)
        axs[1].set_ylabel('Actual value', fontsize=12)
        axs[0].set_ylabel('Probability %', fontsize=12)

        axs[0].annotate(r'$\tilde{\mu}_{{\rm 3, train}}$= ' + f'{np.round(skew_train, 2)}',
                        fontsize=11, xy=(xy_top[0], xy_top[1] + 0.08), xycoords='axes fraction')
        axs[0].annotate(r'$\tilde{\mu}_{{\rm 3, test}}$ = ' + f'{np.round(skew_test, 2)}',
                        fontsize=11, xy=(xy_top[0], xy_top[1] - 0.08), xycoords='axes fraction')

        axs[1].annotate(
            r'R$^2_{\rm train}$=' + f'{"%0.2f" % r_value_train ** 2} [{"%0.2f" % abs(mean_train)}$\pm${"%0.2f" % std_train}]',
            fontsize=11, xy=(xy_bottom[0], xy_bottom[1] + 0.06), xycoords='axes fraction')
        axs[1].annotate(
            r'R$^2_{\rm test}$ =' + f'{np.round(r_value_test ** 2, 2)} [{"%0.2f" % mean_test}$\pm${"%0.2f" % std_test}]',
            fontsize=11, xy=(xy_bottom[0], xy_bottom[1] - 0.06), xycoords='axes fraction')

        axs[1].legend(loc='lower left', fontsize=11)
        # plt.yscale('log')

        f.tight_layout()
        plt.show()            
    
    
    
    
    
    
        
        
        
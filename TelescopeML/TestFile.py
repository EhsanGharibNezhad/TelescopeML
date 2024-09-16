#File to test the Unsupervised_ML.py file with the brown dwarf data from TelescopeML
import numpy as np
import pandas as pd
import seaborn as sns
from TelescopeML.DataMaster import *
from TelescopeML.DeepTrainer import *
from TelescopeML.Predictor import *
from TelescopeML.IO_utils import load_or_dump_trained_model_CNN
from TelescopeML.StatVisAnalyzer import *
import os
from sys import path
import torch
import torch.nn as nn
import Unsupervised_ML
import CNN_Transformer
__reference_data__ = os.getenv("TelescopeML_reference_data")
__reference_data_path__ = __reference_data__




###############################################################################
############################## Data Preparation ###############################
###############################################################################

#Load the data
train_BD = pd.read_csv(os.path.join(__reference_data_path__, 
                                    'training_datasets', 
                                    'browndwarf_R100_v4_newWL_v3.csv.bz2'), compression='bz2')
train_BD.head()
train_BD.shape

#Check the atmospheric parameters
output_names = ['gravity', 'temperature', 'c_o_ratio', 'metallicity']
train_BD[output_names].head()

wavelength_names = [item for item in train_BD.columns.to_list() if item not in output_names]
wavelength_names[:5]

wavelength_values = [float(item) for item in wavelength_names]
wavelength_values[:5]

wl_synthetic = pd.read_csv(os.path.join(__reference_data_path__, 'training_datasets', 'wl.csv'))
wl_synthetic.head()
wl_synthetic.shape


#Prepare the training features X and target variables y for ML models
# Training  variables
X = train_BD.drop(
    columns=['gravity', 
             'temperature', 
             'c_o_ratio', 
             'metallicity'])
X.shape

# Target/Output feature variables
y = train_BD[['gravity', 'c_o_ratio', 'metallicity', 'temperature', ]]
y.shape

#Let's look at the distributions of the target variables
sns.histplot(y['gravity']).set(title='Histogram of gravity')
sns.histplot(y['c_o_ratio']).set(title='Histogram of c_o_ratio')
sns.histplot(y['metallicity']).set(title='Histogram of metallicity')
sns.histplot(y['temperature']).set(title='Histogram of temperature')
#What are the unique number of entries fr these features?
#It seems that temperature may be the only one that should be treated as a continuous feature
y['gravity'].nunique() #11 unique values of gravity
y['c_o_ratio'].nunique() #6 unique values of carbon-oxygen
y['metallicity'].nunique() #12 unique values of metallicity
y['temperature'].nunique() #39 unique values of temperature

#Apply a log transform to the temperature feature since this is what is recommended by the Notebooks
y.loc[:, 'temperature'] = np.log10(y['temperature'])
sns.histplot(y['temperature']).set(title='Histogram of log(temperature)')


# #Let's visualize the brown dwarf spectra for different parameters
# #First: Temperature
# # Define the filter bounds
# filter_bounds = {'gravity': (5.,5), 
#                  'c_o_ratio' : (1,1),
#                  'metallicity' : (0.0,0.0),
#                  'temperature': (400, 1800)}
# # Call the function to filter the dataset
# plot_filtered_spectra(dataset = train_BD, 
#                     filter_bounds = filter_bounds,
#                     feature_to_plot = 'temperature',
#                     title_label = '[log$g$='+str(filter_bounds['gravity'][0])+
#                                   ', C/O ='+str(filter_bounds['c_o_ratio'][0])+
#                                   ', [M/H]='+str(filter_bounds['metallicity'][0])+']',
#                     wl_synthetic = wavelength_values,
#                     output_names = output_names,
#                     __reference_data__ = __reference_data_path__)

# #Second: Gravity
# # Define the filter bounds
# filter_bounds = {'gravity': (3,5.5), 
#                  'c_o_ratio' : (1,1),
#                  'metallicity' : (0.0,0.0),
#                  'temperature': (800, 800)}
# # Call the function to filter the dataset
# plot_filtered_spectra(dataset = train_BD, 
#                         filter_bounds = filter_bounds,
#                         feature_to_plot = 'gravity',
#                         title_label =   '[T='+str(filter_bounds['temperature'][0])+
#                                         ', C/O ='+str(filter_bounds['c_o_ratio'][0])+
#                                         ', [M/H]='+str(filter_bounds['metallicity'][0])+']',
#                         wl_synthetic = wavelength_values,
#                         output_names = output_names,
#                         __reference_data__ = __reference_data_path__)

# #Third: Carbon-to-Oxygen Ratio
# # Define the filter bounds
# filter_bounds = {'gravity': (5.,5), 
#                  'c_o_ratio' : (0.25,2.5),
#                  'metallicity' : (0.0,0.0),
#                  'temperature': (800, 800)}
# # Call the function to filter the dataset
# plot_filtered_spectra(dataset = train_BD, 
#                         filter_bounds = filter_bounds,
#                         feature_to_plot = 'c_o_ratio',
#                         title_label = '[T='+str(filter_bounds['temperature'][0])+
#                                     ', log$g$='+str(filter_bounds['gravity'][0])+
#                                     ', [M/H]='+str(filter_bounds['metallicity'][0])+']',
#                         wl_synthetic = wavelength_values,
#                         output_names = output_names,
#                         __reference_data__ = __reference_data_path__)

# #Fourth: Metallicity
# # Define the filter bounds
# filter_bounds = {'gravity': (5.,5), 
#                  'c_o_ratio' : (1.,1.),
#                  'metallicity' : (-1,2),
#                  'temperature': (800, 800)}
# # Call the function to filter the dataset
# plot_filtered_spectra(dataset = train_BD, 
#                         filter_bounds = filter_bounds,
#                         feature_to_plot = 'metallicity',
#                         title_label = '[T='+str(filter_bounds['temperature'][0])+
#                                 ', log$g$='+str(filter_bounds['gravity'][0])+
#                                 ', C/O='+str(filter_bounds['c_o_ratio'][0])+']',
#                         wl_synthetic = wavelength_values,
#                         output_names = output_names,
#                         __reference_data__ = __reference_data_path__)






###############################################################################
#Using the DataMaster.py module, load the X and y dataframes into a DataProcessor object which has built-in engineering functionality
data_processor = DataProcessor(flux_values=X.to_numpy(),
                               wavelength_names=X.columns,
                               wavelength_values=wavelength_values,
                               output_values=y.to_numpy(),
                               output_names=output_names,
                               spectral_resolution=200)

#Set up training, testing, and validation datasets
data_processor.split_train_validation_test(test_size=0.1, val_size=0.1, random_state_=42)

data_processor.X_train.shape

#Let's look at a boxplot of the original wavelength features
plot_boxplot(data = X.to_numpy(),
            title='Original 104 Wavelength Features',
            xlabel='Wavelength [$\mu$m]',
            ylabel='Values',
            xticks_list=wavelength_names[::-1],
            fig_size=(18, 5),
            saved_file_name = 'Input_Fluxes',
            __reference_data__ = __reference_data_path__)

# Scale (standardize) the X features using MinMax Scaler
data_processor.standardize_X_row_wise()

#Let's look at a boxplot of the standardized wavelength features
plot_boxplot(data = data_processor.X_train_standardized_rowwise[:, ::-1],
            title='Standardized Wavelength 104 Features',
            xlabel='Wavelength [$\mu$m]',
            ylabel='Scaled Values',
            xticks_list=wavelength_names[::-1],
            fig_size=(18, 5),
            saved_file_name = 'Scaled_input_fluxes',
            __reference_data__ = __reference_data_path__)


# Standardize the y features using Standard Scaler
data_processor.standardize_y_column_wise()

#Let's look at a boxplot of the standardized target features
plot_boxplot(data = data_processor.y_train_standardized_columnwise,
            title='Scaled 4 Target Features',
            xlabel='Wavelength',
            ylabel='Scaled Output Values',
            xticks_list=['','$\log g$', 'T$_{eff}$', 'C/O ratio', '[M/H]'],
            fig_size=(5, 5),
            saved_file_name = 'Scaled_output_parameters',
            __reference_data__ = __reference_data_path__)


#Apply the same feature engineering to take the min and max of each row
# train
data_processor.X_train_min = data_processor.X_train.min(axis=1)
data_processor.X_train_max = data_processor.X_train.max(axis=1)
# validation
data_processor.X_val_min = data_processor.X_val.min(axis=1)
data_processor.X_val_max = data_processor.X_val.max(axis=1)
# test
data_processor.X_test_min = data_processor.X_test.min(axis=1)
data_processor.X_test_max = data_processor.X_test.max(axis=1)

df_MinMax_train = pd.DataFrame((data_processor.X_train_min, data_processor.X_train_max)).T
df_MinMax_val = pd.DataFrame((data_processor.X_val_min, data_processor.X_val_max)).T
df_MinMax_test = pd.DataFrame((data_processor.X_test_min, data_processor.X_test_max)).T

df_MinMax_train.rename(columns={0:'min', 1:'max'}, inplace=True)
df_MinMax_val.rename(columns={0:'min', 1:'max'}, inplace=True)
df_MinMax_test.rename(columns={0:'min', 1:'max'}, inplace=True)

df_MinMax_train.head()

#Now Standardize the min and max features
data_processor.standardize_X_column_wise(output_indicator='Trained_StandardScaler_X_ColWise_MinMax',
                                        X_train = df_MinMax_train.to_numpy(),
                                        X_val = df_MinMax_val.to_numpy(),
                                        X_test = df_MinMax_test.to_numpy())


#Concatenate all of the features into training, testing, and validation dataframes
X_train_standardized = pd.concat([pd.DataFrame(data_processor.X_train_standardized_rowwise), pd.DataFrame(data_processor.X_train_standardized_columnwise)],axis=1)
X_val_standardized = pd.concat([pd.DataFrame(data_processor.X_val_standardized_rowwise), pd.DataFrame(data_processor.X_val_standardized_columnwise)],axis=1)
X_test_standardized = pd.concat([pd.DataFrame(data_processor.X_test_standardized_rowwise), pd.DataFrame(data_processor.X_test_standardized_columnwise)],axis=1)





###############################################################################
############################## Unsupervised_ML.py #############################
###############################################################################

#Instantiate an UnsupervisedML object
UML = Unsupervised_ML.UnsupervisedML(X_train_standardized)

#Generate the PCA embeddings
UML.pca(train_data=UML.dataset, val_data=None, test_data=None, n_components=10, random_state=42)

#Generate k-Means elbow plot
UML.kmeans_elbow_plot(train_data=UML.pca_train_dataset, max_k = 20, n_init = 50, max_iter=500, random_state=42)

#Generate K-means clusters using the PCA embeddings
UML.kmeans(train_data=UML.pca_train_dataset, val_data=None, test_data=None, n_clusters = 15, n_init = 50, max_iter=500, random_state=42)
UML.scatter_3d_cluster_plot(dataset=UML.train_kmeans_clustered, method='PCA_KMeans')

#Generate DBSCAN clusters using the PCA embeddings
UML.dbscan(train_data=UML.pca_train_dataset, val_data=None, test_data=None, eps=0.5, min_samples=10, algorithm='brute', p=2)
UML.scatter_3d_cluster_plot(dataset=UML.train_dbscan_clustered, method='PCA_DBSCAN')

#Generate spectral clusters using the PCA embeddings
UML.spec_c(train_data=UML.pca_train_dataset, affinity_metric='rbf', n_clusters=8, assign_labels="discretize", random_state=42)
UML.scatter_3d_cluster_plot(dataset=UML.train_spec_c_clustered, method='PCA_Spec_c')

#Generate t-SNE embeddings of the PCs
UML.tsne(train_data=UML.pca_train_dataset, val_data=None, test_data=None, n_components = 3, learning_rate='auto', perplexity=200, random_state=42)

#Generate K-means clusters using the t-SNE embeddings
UML.kmeans(train_data=UML.tsne_train_dataset, val_data=None, test_data=None, n_clusters = 8, n_init = 50, max_iter=500, random_state=42)
UML.scatter_3d_cluster_plot(dataset=UML.train_kmeans_clustered, method='TSNE_KMeans')

#Generate DBSCAN clusters using the t-SNE embeddings
UML.dbscan(train_data=UML.tsne_train_dataset, val_data=None, test_data=None, eps=0.5, min_samples=10, algorithm='brute', p=2)
UML.scatter_3d_cluster_plot(dataset=UML.train_dbscan_clustered, method='TSNE_DBSCAN')

#Generate spectral clusters using the t-SNE embeddings
UML.spec_c(train_data=UML.tsne_train_dataset, affinity_metric='rbf', n_clusters=8, assign_labels="discretize", random_state=42)
UML.scatter_3d_cluster_plot(dataset=UML.train_spec_c_clustered, method='TSNE_Spec_c')





###############################################################################
############################## CNN_Transformer.py #############################
###############################################################################

#Set the device for pytorch to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Instantiate a new CNNTransformer object
CNNT_Modeler = CNN_Transformer.CNNTransformer()
CNN_Transformer.Set_Seed(42)
CNNT_Modeler.get_loaders(0.01)
model = CNNT_Modeler.build_cnntransformer_model()
loss_func = nn.HuberLoss().to(device) #Define the corss-entropy loss as our function that we are tring to minize
epochs = 10
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0) #Use SGD as the optimization algorithm
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=epochs, steps_per_epoch=len(CNNT_Modeler.train_loader))
CNNT_Modeler.deep_train(model, loss_func, CNNT_Modeler.train_loader, CNNT_Modeler.val_loader, epochs, lr_scheduler, optimizer)
fig = CNNT_Modeler.plot_losses()


#Or simply load the one that was already trained
CNNT_Modeler = CNN_Transformer.CNNTransformer()
path = 'cnntransformer_trained_10epoch.pth'
CNNT_Modeler.load_model(path)




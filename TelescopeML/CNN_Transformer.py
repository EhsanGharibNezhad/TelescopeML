#Bonus: CNN & Transformer Model for TelescopeML Spectra Data
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from TelescopeML.DataMaster import *
from TelescopeML.DeepTrainer import *
from TelescopeML.Predictor import *
from TelescopeML.IO_utils import load_or_dump_trained_model_CNN
from TelescopeML.StatVisAnalyzer import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import random
import pickle
__reference_data__ = os.getenv("TelescopeML_reference_data")
__reference_data_path__ = __reference_data__



#Set the device for pytorch to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############################################################################
############################## Utility Functions ##############################
###############################################################################

#Function to set the seed
def Set_Seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

class Permute(nn.Module):
    def forward(self, x):
        # x shape: (batch_size, num_channels, sequence_length)
        return x.permute(2, 0, 1)

class MeanAggregation(nn.Module):
    def forward(self, x):
        # x shape: (sequence_length, batch_size, num_features)
        return x.mean(dim=0)  # Output shape: (batch_size, num_features)

class SpectraDataset(Dataset):
    def __init__(self, X, y):
        self.data = X.values  # Features (all columns except the last one)
        self.targets = y.values  # Targets (the last column)
        
    def __len__(self):
        return len(self.data)  # Number of samples in the dataset
    
    def __getitem__(self, index):
        # Get the features and target for the given index
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.targets[index], dtype=torch.float32)
        x = x.unsqueeze(0)
        return x, y


###############################################################################
############################ CNNTransformer Class #############################
###############################################################################

class CNNTransformer(nn.Module):
    def __init__(self):
        super(CNNTransformer, self).__init__()
        #Initialize the data
        self.data_processor = self.load_spectra_data()
        
        self.X_train = pd.concat([pd.DataFrame(self.data_processor.X_train_standardized_rowwise), pd.DataFrame(self.data_processor.X_train_standardized_columnwise)],axis=1)
        self.X_val = pd.concat([pd.DataFrame(self.data_processor.X_val_standardized_rowwise), pd.DataFrame(self.data_processor.X_val_standardized_columnwise)],axis=1)
        self.X_test = pd.concat([pd.DataFrame(self.data_processor.X_test_standardized_rowwise), pd.DataFrame(self.data_processor.X_test_standardized_columnwise)],axis=1)
        
        self.y_train = pd.DataFrame(self.data_processor.y_train_standardized_columnwise, columns = self.data_processor.output_names)
        self.y_val = pd.DataFrame(self.data_processor.y_val_standardized_columnwise, columns = self.data_processor.output_names)
        self.y_test = pd.DataFrame(self.data_processor.y_test_standardized_columnwise, columns = self.data_processor.output_names)



    def load_spectra_data(self):
        """
        Executes the functionality to load and prepare (standardize & feature engineer) the spectra data from the notebooks.
        
        """
        #Load the data
        train_BD = pd.read_csv(os.path.join(__reference_data_path__, 'training_datasets', 'browndwarf_R100_v4_newWL_v3.csv.bz2'), compression='bz2')
        output_names = ['gravity', 'temperature', 'c_o_ratio', 'metallicity']
        wavelength_names = [item for item in train_BD.columns.to_list() if item not in output_names]
        wavelength_values = [float(item) for item in wavelength_names]
        #Prepare the training features X and target variables y for ML models
        # Training  variables
        X = train_BD.drop(
            columns=['gravity', 
                     'temperature', 
                     'c_o_ratio', 
                     'metallicity'])
        # Target/Output feature variables
        y = train_BD[['gravity', 'c_o_ratio', 'metallicity', 'temperature', ]]
        #Apply a log transform to the temperature feature since this is what is recommended by the Notebooks
        y.loc[:, 'temperature'] = np.log10(y['temperature'])
        
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
        # Scale (standardize) the X features using MinMax Scaler
        data_processor.standardize_X_row_wise()
        # Standardize the y features using Standard Scaler
        data_processor.standardize_y_column_wise()
        
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
        
        #Now Standardize the min and max features
        data_processor.standardize_X_column_wise(output_indicator='Trained_StandardScaler_X_ColWise_MinMax',
                                                X_train = df_MinMax_train.to_numpy(),
                                                X_val = df_MinMax_val.to_numpy(),
                                                X_test = df_MinMax_test.to_numpy())
    
        # #Concatenate all of the features into training, testing, and validation dataframes
        # X_train_standardized = pd.concat([pd.DataFrame(data_processor.X_train_standardized_rowwise), pd.DataFrame(data_processor.X_train_standardized_columnwise)],axis=1)
        # X_val_standardized = pd.concat([pd.DataFrame(data_processor.X_val_standardized_rowwise), pd.DataFrame(data_processor.X_val_standardized_columnwise)],axis=1)
        # X_test_standardized = pd.concat([pd.DataFrame(data_processor.X_test_standardized_rowwise), pd.DataFrame(data_processor.X_test_standardized_columnwise)],axis=1)
        
        # y_train = pd.DataFrame(data_processor.y_train_standardized_columnwise, columns = data_processor.output_names)
        # y_val = pd.DataFrame(data_processor.y_test_standardized_columnwise, columns = data_processor.output_names)
        # y_test = pd.DataFrame(data_processor.y_val_standardized_columnwise, columns = data_processor.output_names)
        
        return data_processor#, X_train_standardized, X_val_standardized, X_test_standardized, y_train, y_val, y_test

    

    def build_cnntransformer_model(self):
        """
        Builds the model composed of convolution layers followed by transformer encoder layers.
        
        """
        #Define the Transformer Layer that will be utilized (The paremeters of this model were obtained from the final CNN layer)
        transformer_layer = nn.TransformerEncoderLayer(
                                                       d_model=512, #number of cnn channels
                                                       nhead=8, #needs to divide 512
                                                       dim_feedforward=512*4, #The size of the feedforward network within the transformer. Typically set to 2 to 4 times the d_model
                                                       activation='relu'
                                                       )
        #The block of transformer layers; we will use 1
        transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
        
        #Define the full CNN-Transformer model
        cnntransformer_model = nn.Sequential(
                                             nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1), #Turns the [1,104] input into [32,104]
                                             nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, padding=1), #Turns the [32,104] input into [128,104]
                                             nn.Conv1d(in_channels=128, out_channels=288, kernel_size=3, padding=1), #Turns the [128,104] input into [228,104]
                                             nn.MaxPool1d(kernel_size=3, stride=3), #Turns the [228,104] input into [228,34]
                                             nn.Conv1d(in_channels=288, out_channels=512, kernel_size=3, padding=1), #Turns the [228,34] input into [512,34]
                                             nn.MaxPool1d(kernel_size=3, stride=3), #Turns the [512,34] input into [512,11]
                                             Permute(), #Alters the input to 
                                             transformer,
                                             MeanAggregation(),
                                             nn.Linear(512, 4)
                                             )
        self.cnntransformer_model = cnntransformer_model
        return cnntransformer_model
    
    
    def get_loaders(self, batch_size_percent=0.001):
        """
        Function that builds the pre-batched data loaders for the training, validation, and testing datasets.
        
        Parameters
        ----------
        batch_size_percent : float
            Percentage of the datasets to use to split into batches.
        
        """
        #Create the dataset instances
        train_dataset = SpectraDataset(self.X_train, self.y_train)
        val_dataset = SpectraDataset(self.X_val, self.y_val)
        test_dataset = SpectraDataset(self.X_test, self.y_test)
        
        #Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=int(np.floor(self.X_train.shape[0]*batch_size_percent)), shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=int(np.floor(self.X_val.shape[0]*batch_size_percent)), shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=int(np.floor(self.X_test.shape[0]*batch_size_percent)), shuffle=True)
        return 

        
    def deep_train(self, model, loss_func, train_loader, val_loader, epochs, lr_scheduler, optimizer):
        """
        The algorithm to train the deep learning model.
        
        Parameters
        ----------
        model : torch.nn
            The deep learning model.
        loss_func : torch.nn.F
            The objective function that we are trying to minimize.
        train_loader : torch.utils.data.DataLoader
            The pre-batched loader for the training dataset.
        val_loader : torch.utils.data.DataLoader
            The pre-batched loader for the validation dataset.
        epochs : int
            The number of passes through the data loaders.
        lr_scheduler : torch.optim.lr_scheduler
            The scheduler that updates the learning rate.
        optimizer : torch.optim
            The optimization algorithm used to update the model parameters.
        
        """
        #Data collection
        total_train_loss_list = []; total_val_loss_list = []
        gravity_train_loss_list = []; gravity_val_loss_list = []
        temp_train_loss_list = []; temp_val_loss_list = []
        c_o_train_loss_list = []; c_o_val_loss_list = []
        met_train_loss_list = []; met_val_loss_list = []
        
        #Iterate through the number of epochs
        j = 0
        for epoch in range(epochs):
            print(epoch)
            #For each epoch, iterate through all of the training batches
            i=0
            for X_tr, y_tr in train_loader:
                print(i)
                i+=1
                X_tr = X_tr.to(device); y_tr = y_tr.to(device)
                #Update the variables with a single step
                model.train() #Puts the model into the "training" mode (makes sure to track the gradients)
                optimizer.zero_grad() #Zeros out the gradients from the previous update
                pred = model(X_tr) #Use the model to predict the outputs of the batch
                total_loss = loss_func(pred, y_tr) #Compute the error of the prediction
                total_train_loss_list.append(float(total_loss))
                total_loss.backward() #Using the error, backpropigate to compute the gradients
                optimizer.step() #Using the gradients that were computed, update the model with a single step via the optimizer that was chosen
                lr_scheduler.step()
                j += 1
                
                model.eval()
                with torch.no_grad():
                    pred0 = pred.T[0]; pred1 = pred.T[1]; pred2 = pred.T[2]; pred3 = pred.T[3]
                    y0 = y_tr.T[0]; y1 = y_tr.T[1]; y2 = y_tr.T[2]; y3 = y_tr.T[3]
                    loss0 = loss_func(pred0, y0); loss1 = loss_func(pred1, y1); loss2 = loss_func(pred2, y2); loss3 = loss_func(pred3, y3)
                    gravity_train_loss_list.append(float(loss0)); temp_train_loss_list.append(float(loss1)); c_o_train_loss_list.append(float(loss2)); met_train_loss_list.append(float(loss3))
                    #Compute the loss on the testing dataset
                    X_test, y_test = next(iter(val_loader))
                    X_test = X_test.to(device); y_test = y_test.to(device)
                    val_pred = model(X_test)
                    val_loss = loss_func(val_pred, y_test)
                    total_val_loss_list.append(float(val_loss))
                    pred0 = val_pred.T[0]; pred1 = val_pred.T[1]; pred2 = val_pred.T[2]; pred3 = val_pred.T[3]
                    y0 = y_test.T[0]; y1 = y_test.T[1]; y2 = y_test.T[2]; y3 = y_test.T[3]
                    loss0 = loss_func(pred0, y0); loss1 = loss_func(pred1, y1); loss2 = loss_func(pred2, y2); loss3 = loss_func(pred3, y3)
                    gravity_val_loss_list.append(float(loss0)); temp_val_loss_list.append(float(loss1)); c_o_val_loss_list.append(float(loss2)); met_val_loss_list.append(float(loss3))
                    
                
        self.total_train_loss_list = total_train_loss_list
        self.total_val_loss_list = total_val_loss_list
        self.gravity_train_loss_list = gravity_train_loss_list
        self.gravity_val_loss_list = gravity_val_loss_list
        self.temp_train_loss_list = temp_train_loss_list
        self.temp_val_loss_list = temp_val_loss_list
        self.c_o_train_loss_list = c_o_train_loss_list
        self.c_o_val_loss_list = c_o_val_loss_list
        self.met_train_loss_list = met_train_loss_list
        self.met_val_loss_list = met_val_loss_list
        return [total_train_loss_list, total_val_loss_list]
    
    
    def plot_losses(self):
        """
        Function to plot the loss function from training the model.
        
        """
        train_loss = self.total_train_loss_list
        val_loss = self.total_val_loss_list
        gravity_train = self.gravity_train_loss_list
        gravity_val = self.gravity_val_loss_list
        temp_train = self.temp_train_loss_list
        temp_val = self.temp_val_loss_list
        c_o_train = self.c_o_train_loss_list
        c_o_val = self.c_o_val_loss_list
        met_train = self.met_train_loss_list
        met_val = self.met_val_loss_list
        #Plot the losses
        iter_list = [i for i in range(len(train_loss))]
        fig, axes = plt.subplots(5, 1, figsize=(6, 8), sharex=True)
        sns.lineplot(x=iter_list, y=train_loss, ax=axes[0], linewidth=0.3, label='Total Training Loss', legend=True, color='red', linestyle='-').set_title('Total Loss: Training Deep CNN-Transformer Network', size=15)
        sns.lineplot(x=iter_list, y=val_loss, ax=axes[0], linewidth=0.3, label='Total Validation Loss', color='red', linestyle='--')
        sns.lineplot(x=iter_list, y=gravity_train, ax=axes[1], linewidth=0.3, label='$log g$ Training Loss', color='blue', linestyle='-').set_title('Gravity Loss: Training Deep CNN-Transformer Network', size=15)
        sns.lineplot(x=iter_list, y=gravity_val, ax=axes[1], linewidth=0.3, label='$log g$ Validation Loss', color='blue', linestyle='--')
        sns.lineplot(x=iter_list, y=temp_train, ax=axes[2], linewidth=0.3, label='$T_{eff}$ Training Loss', color='green', linestyle='-').set_title('Temperature Loss: Training Deep CNN-Transformer Network', size=15)
        sns.lineplot(x=iter_list, y=temp_val, ax=axes[2], linewidth=0.3, label='$T_{eff}$ Validation Loss', color='green', linestyle='--')
        sns.lineplot(x=iter_list, y=c_o_train, ax=axes[3], linewidth=0.3, label='$[M/H]$ Training Loss', color='orange', linestyle='-').set_title('Metallicity Loss: Training Deep CNN-Transformer Network', size=15)
        sns.lineplot(x=iter_list, y=c_o_val, ax=axes[3], linewidth=0.3, label='$[M/H]$ Validation Loss', color='orange', linestyle='--')
        sns.lineplot(x=iter_list, y=met_train, ax=axes[4], linewidth=0.3, label='$C/O$ Training Loss', color='purple', linestyle='-').set_title('C/O Ratio Loss: Training Deep CNN-Transformer Network', size=15)
        sns.lineplot(x=iter_list, y=met_val, ax=axes[4], linewidth=0.3, label='$C/O$ Validation Loss', color='purple', linestyle='--')
        plt.tight_layout()
        plt.xlabel("Iterations")
        fig.text(0.001, 0.5, 'Huber Loss', va='center', rotation='vertical', fontsize=12)
        return fig
    
    
    def save_model(self, path):
        """
        Saves a trained model to specified location.
        
        Parameters
        ----------
        path : str
            Save location.
        
        """
        torch.save(self.cnntransformer_model.state_dict(), path)
        return
    
    
    def load_model(self, path):
        """
        Loads a trained model from a specified location.
        
        Parameters
        ----------
        path : str
            Load location.
        
        """
        model = self.build_cnntransformer_model()
        model.load_state_dict(torch.load(path))
        self.cnntransformer_model = model
        return model
    




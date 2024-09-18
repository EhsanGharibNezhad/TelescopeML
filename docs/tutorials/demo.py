# Unsupervised learning models
from TelescopeML.UnsupervisedLearning import *

# Data processing
from TelescopeML.DataMaster import *

# Demo hosting foundation
import streamlit as st

# Data/math imports
import pandas as pd
import numpy as np

# File imports
import os
import tempfile
import io
import zipfile

# =============
# Instructions
# =============
#
# 1. If you haven't already, make sure to run "pip install streamlit" for the necessary dependencies
# 2. In the command prompt, run "streamlit run docs/tutorials/demo.py"
# 3. This should open up a new tab with the demo website. From there you can explore the models of
#    my unsupervised learning implementation through the dropdown menu.



# --------------------------------------
# ---    DATA PROCESSOR FUNCTIONS    ---
# --------------------------------------


def create_data_processor():
    """
    Read CSV data from local reference path, and construct DataProcesor object from it
    """
    
    # read reference data from local CSV file
    __reference_data_path__ = os.getenv("TelescopeML_reference_data")
    train_BD = pd.read_csv(os.path.join(__reference_data_path__, 
                                        'training_datasets', 
                                        'browndwarf_R100_v4_newWL_v3.csv.bz2'), compression='bz2')

    # Training  variables
    X = train_BD.drop(
        columns=['gravity', 
                'temperature', 
                'c_o_ratio', 
                'metallicity'])


    # Target/Output feature variables
    y = train_BD[['gravity', 'c_o_ratio', 'metallicity', 'temperature', ]]
    y.loc[:, 'temperature'] = np.log10(y['temperature'])

    # relevant parameters for DataProcessor
    output_names = ['gravity', 'temperature', 'c_o_ratio', 'metallicity']
    wavelength_names = [item for item in train_BD.columns.to_list() if item not in output_names]
    wavelength_values = [float(item) for item in wavelength_names]

    # construct DataProcessor
    data_processor = DataProcessor( 
                                    flux_values=X.to_numpy(),
                                    wavelength_names=X.columns,
                                    wavelength_values=wavelength_values,
                                    output_values=y.to_numpy(),
                                    output_names=output_names,
                                    spectral_resolution=200,
                                    trained_ML_model=None,
                                    trained_ML_model_name='CNN',
                                    )
    return data_processor


def dp_scale_x_y(data_processor: DataProcessor):
    """
    Perform relevant operations to scale DataProcessor's X & Y data
    column-wise and row-wise as necessary
    """

    data_processor = create_data_processor()
    data_processor.split_train_validation_test(test_size=0.1, 
                                                val_size=0.1, 
                                                random_state_=42,)
    data_processor.standardize_X_row_wise()
    data_processor.standardize_y_column_wise()

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

    data_processor.standardize_X_column_wise(
                                            output_indicator='Trained_StandardScaler_X_ColWise_MinMax',
                                            X_train = df_MinMax_train.to_numpy(),
                                            X_val   = df_MinMax_val.to_numpy(),
                                            X_test  = df_MinMax_test.to_numpy(),
                                            )
    return data_processor


# --------------------------------------
# --------    PCA FUNCTIONS    ---------
# --------------------------------------


def initialize_pca(data_processor) -> PCAModel:
    """
    Build & fit PCA model to X train
    """
    # Build PCA Model
    pca = PCAModel(
        X_train = data_processor.X_train_standardized_rowwise,
        X_val   = data_processor.X_val_standardized_rowwise,
        X_test  = data_processor.X_test_standardized_rowwise,
        y_train = data_processor.y_train_standardized_columnwise,
        y_val   = data_processor.y_val_standardized_columnwise,
        y_test  = data_processor.y_test_standardized_columnwise,
    )
    pca.build_model({'n_components': .999})
    pca.train_model()
    return pca


# --------------------------------------
# ----    AUTOENCODER FUNCTIONS    -----
# --------------------------------------


def initialize_autoencoder(data_processor) -> Autoencoder:
    """
    Build & fit autoencoder model to X train
    """
    # Load existing Autoencoder model
    autoencoder = Autoencoder(
        X_train = data_processor.X_train_standardized_rowwise,
        X_val   = data_processor.X_val_standardized_rowwise,
        X_test  = data_processor.X_test_standardized_rowwise,
        y_train = data_processor.y_train_standardized_columnwise,
        y_val   = data_processor.y_val_standardized_columnwise,
        y_test  = data_processor.y_test_standardized_columnwise,
    )
    config = {
        'output_dim': 20,
        'encoder_layers': [60, 35],
        'decoder_layers': [35, 60],
        # Exclude final input & output layer for each
    }
    autoencoder.build_model(config)
    return autoencoder



# --------------------------------------
# --------    DEMO HOME PAGE    --------
# --------------------------------------

"""
# Unsupervised Learning Visualized

Abhinav Uppala

*For each of these examples, we use a random test dataset value to show model's effectiveness. For more concrete results,
download the model files and use the UnsupervisedLearning module from TelescopeML to load, train and test these models.*
"""

# Initialize DataProcessor object
data_processor = create_data_processor()
data_processor = dp_scale_x_y(data_processor)


# Dropdown
vis_dropdown = st.selectbox(label='Pick visualization',
                            options=['PCA', 'Autoencoder', 'GMM'],
                            index=None,
                            key='visualization')


# --------------------------------------
# -----    DEMO DROPDOWN PAGES    ------
# --------------------------------------

if st.session_state.get('visualization'):
    
    # --------------------------------------
    # -------------    PCA    --------------
    # --------------------------------------

    if vis_dropdown == 'PCA':
        """
        ## PCA for Dimensionality Reduction

        PCA (Principal Component Analysis) is a method of reducing the input's dimensions, while also preserving important features and variation in the data.
        This specific model aims to preserve at least 99.9% of the variation in the training data. This demo shows its effectiveness by comparing a
        random test X value, along with the same PCA-compressed & reconstructed X value, demonstrating how features are preserved.
        """
        with st.spinner('Initializing PCA...'):
            pca = initialize_pca(data_processor)
        index = np.random.randint(data_processor.X_test_standardized_rowwise.shape[0])

        actual = data_processor.X_test_standardized_rowwise[index].reshape(1, -1)

        encoded = pca.encode_input_data(actual)
        estimated = pca.decode_output_data(encoded)

        st.line_chart(pd.DataFrame(
            {'Actual': actual[0], 'Reconstructed': estimated[0]}
        ))

    
    # --------------------------------------
    # ---------    AUTOENCODER    ----------
    # --------------------------------------

    elif vis_dropdown == 'Autoencoder':
        """
        ### Autoencoder for Dimensionality Reduction

        Autoencoders are models with 2 parts - encoder & decoder, used to transform an input into a different state and transform it back. Training the encoder and
        decoder ensures the model is able to identify trends and variability in the original data and preseve them in a different format. By encoding our input data into
        a lower dimension (104 -> 20 in this case), we can perform dimensionality reduction while retaining the original features of the input. This demo showcases this
        by showing a randomly selected X value along with it's reconstruction from encoding & decoding.
        """
        # Allow users to upload their own weights or train a new model
        # because users who download the TelescopeML_project file won't have the
        # model from my local machine
        file_dropdown = st.selectbox(label='How to create model',
                                     options=['Upload weights files', 'Train new model'],
                                     index=None)

        with st.spinner('Initializing Autoencoder...'):
            autoencoder = initialize_autoencoder(data_processor)
        index = np.random.randint(data_processor.X_test_standardized_rowwise.shape[0])

        actual = data_processor.X_test_standardized_rowwise[index].reshape(1, -1)

        # Compare actual & reconstructed before training
        encoded_no_train = autoencoder.encoder.predict(actual)
        estimated_no_train = autoencoder.decoder.predict(encoded_no_train)

        # keep track of if the model has been trained
        # if so, we plot the model twice - before & after
        # otherwise, we just plot the before
        model_trained = False

        if file_dropdown == 'Upload weights files':
            
            # User input for weights files
            with st.form('File Upload'):
                encoder_weights = st.file_uploader('Encoder Weights')
                decoder_weights = st.file_uploader('Decoder Weights')
                submitted = st.form_submit_button('Submit')

            if submitted:

                # file names (same as uploaded) to be used in ZIP download
                encoder_name = encoder_weights.name.rstrip('h5')
                decoder_name = decoder_weights.name.rstrip('h5')

                # convert bytes to temporary file object
                encoder_bytes = encoder_weights.getvalue()
                decoder_bytes = decoder_weights.getvalue()

                # load encoder weights
                with tempfile.NamedTemporaryFile(suffix='.h5', mode='wb+', delete=False) as f:
                    f.write(encoder_bytes)
                    f.flush()
                    encoder_filename = f.name

                # load decoder weights
                with tempfile.NamedTemporaryFile(suffix='.h5', mode='wb+', delete=False) as f:
                    f.write(decoder_bytes)
                    f.flush()
                    decoder_filename = f.name
                    
                # load weights from tempfiles
                autoencoder.encoder.load_weights(encoder_filename)
                autoencoder.decoder.load_weights(decoder_filename)

                # delete tempfiles
                # files must be deleted seperately from the with statement
                # because loading weights while in write more is not possible
                os.remove(encoder_filename)
                os.remove(decoder_filename)

                model_trained = True

        elif file_dropdown == 'Train new model':

            # User input for weights files
            with st.form('Training Parameters'):
                epochs = st.slider(label='Epochs',
                                   min_value=1, max_value=200, value=25)
                batch_size = st.slider(label='Batch Size',
                                    min_value=1, max_value=data_processor.X_train.shape[0], value=50)
                submitted = st.form_submit_button('Submit')

                # by default, model name is based on parameters but can be modified.
                # used for deciding filename of the download button
                encoder_name = st.text_input(label='Choose your model name',
                                        value=f'trained_encoder_{epochs}-epoch_{batch_size}-batch',
                                        max_chars=100)
                decoder_name = st.text_input(label='Choose your model name',
                                        value=f'trained_decoder_{epochs}-epoch_{batch_size}-batch',
                                        max_chars=100)

            # train the model
            if submitted:
                with st.spinner(text='Training...'):
                    model_history, _ = autoencoder.train_model(epochs=epochs, batch_size=batch_size, verbose=1)
                model_trained = True


        if not model_trained:
            st.line_chart(pd.DataFrame(
            {'Actual': actual[0],
             'Reconstructed (Before Training)': estimated_no_train[0],}
            ))

        else:
            # Compare actual & reconstructed after training
            encoded = autoencoder.encoder.predict(actual)
            estimated = autoencoder.decoder.predict(encoded)

            st.line_chart(pd.DataFrame(
                {'Actual': actual[0],
                'Reconstructed (Before Training)': estimated_no_train[0],
                'Reconstructed (After Training)': estimated[0]}
            ))

            # show loss if model history exists
            if file_dropdown == 'Train new model':
                f"""
                Loss: {model_history.history['loss'][-1]}
                """

            # open multiple tempfiles
            with tempfile.NamedTemporaryFile(suffix='.h5', mode='wb+', delete=False) as ef, \
                 tempfile.NamedTemporaryFile(suffix='.h5', mode='wb+', delete=False) as df:
                
                encoder_file = ef.name
                decoder_file = df.name

            # save weights to the given tempfiles
            autoencoder.save_weights(encoder_file, decoder_file)

            # since download button refreshes page, meaning model will have to be reconstructed
            #   we want to download it all at once. Since download_button only supports 1 at a time,
            #   we must pack them into a ZIP file to avoid unnecessary re-training

            # read bytes data from saved weights temp files
            with open(encoder_file, 'rb') as ef, open(decoder_file, 'rb') as df:
                encoder_data = ef.read()
                decoder_data = df.read()

            # pack our 2 files into a ZIP, resetting buffer position
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr(f'{encoder_name}.h5', encoder_data)
                zip_file.writestr(f'{decoder_name}.h5', decoder_data)
            zip_buffer.seek(0)

            # create download button for ZIP
            st.download_button(
                label="Download Encoder and Decoder Models",
                data=zip_buffer,
                file_name="models.zip",
                mime="application/zip"
            )
            
            # TODO: implement download weights button with TF .h5 format

    # --------------------------------------
    # -------------    GMM    --------------
    # --------------------------------------

    elif vis_dropdown == 'GMM':
        """
        ### GMMs for Clustering

        GMMs (Gaussian Mixture Models) represent a dataset as a mix of several gaussian distributions, each corresponding to a cluster. The reason I considered
        GMMs initially over another clustering model like K-means is because K-means uses hard clustering (each point belongs to one cluster only) while GMMs use soft
        clustering (each point has a probability of being in each cluster), and shapes other than just spheres. However, one big drawback is how slow GMMs are to train.
        Here, we use PCA to encode our input into clusterable dimensions. To try autoencoders, use these modules in your code.
        """
        # Allow users to upload their own weights or train a new model
        # because it takes a long time to fit GMM
        file_dropdown = st.selectbox(label='How to create model',
                                     options=['Upload model files', 'Train (Not recommended)'],
                                     index=None)
        gmm_trained = False
        
        # Don't load PCA and gmm if nothing is selected
        if file_dropdown is not None:

            with st.spinner('Initializing PCA...'):
                pca = initialize_pca(data_processor)
            index = np.random.randint(data_processor.X_test_standardized_rowwise.shape[0])

            # use PCA for dimensionality reduction (for now at least)
            # because we can store locally rather than needing to store/load an autoencoder
            actual = data_processor.X_test_standardized_rowwise[index].reshape(1, -1)
            encoded = pca.encode_input_data(actual)

            with st.spinner('Loading GMM...'):
                gmm = GMM(
                    X_train = pca.encode_input_data(data_processor.X_train_standardized_rowwise),
                    X_val   = pca.encode_input_data(data_processor.X_val_standardized_rowwise),
                    X_test  = pca.encode_input_data(data_processor.X_test_standardized_rowwise),
                    y_train = data_processor.y_train_standardized_columnwise,
                    y_val   = data_processor.y_val_standardized_columnwise,
                    y_test  = data_processor.y_test_standardized_columnwise,
                )
        
        # upload model .pkl files
        if file_dropdown == 'Upload model files':

            # File user input
            with st.form('File Upload'):
                gmm_upload = st.file_uploader('GMM .pkl File')
                submitted = st.form_submit_button('Submit')

            if submitted:

                # load as bytes object
                gmm_name = gmm_upload.name.rstrip('.pkl')
                gmm_bytes = gmm_upload.getvalue()

                # load into temporary file
                with tempfile.NamedTemporaryFile(suffix='.pkl', mode='wb+', delete=False) as f:
                    f.write(gmm_bytes)
                    f.flush()
                    gmm_filename = f.name

                # load weights from tempfiles
                gmm.load_from_path(gmm_filename)
                gmm_trained = True

                # delete tempfile
                os.remove(gmm_filename)
            

        elif file_dropdown == 'Train (Not recommended)':
            """
            NOTE: With such a small traning amount, the model will not be very effective. To see a fully effective model, upload your own model pkl file.
            Training for longer max iterations and components will take much longer, but will lead to better results, as an alternative to uploading.
            """
            # User input for training parameters
            with st.form('Training Parameters'):
                max_iter = st.slider(label='Max Iterations',
                                   min_value=1, max_value=100, value=10)
                components = st.slider(label='Components',
                                    min_value=1, max_value=50, value=15)
                covariance_type = st.selectbox(
                    label='Covariance Type', options=['full', 'tied', 'diag', 'spherical'], index=0
                )

                # by default, model name is based on parameters but can be modified.
                # used for deciding filename of the download button
                gmm_name = st.text_input(label='Choose your model name',
                                        value=f'trained_gmm_{components}-component_{covariance_type}-covar_{max_iter}-iter',
                                        max_chars=100)
                
                submitted = st.form_submit_button('Submit')

            if submitted:
                with st.spinner('Training GMM...'):
                    config = {
                        'n_components' : components,            # Cluster count
                        'covariance_type' : covariance_type,    # full, tied, spherical, diag
                        'max_iter' : max_iter,                  # Max EM steps to do
                        'n_init' : 1,                           # Number of initializations to make
                        'verbose': 1,
                    }
                    gmm.build_model(config)
                    gmm.train_model()
                    gmm_trained = True
            
        # Again, only plot if the GMM has been trained
        if gmm_trained:
            predicted_y = gmm.predict(encoded)
            actual_y = data_processor.y_test[index]

            st.bar_chart(pd.DataFrame(
                {'Actual': actual_y,
                'Predicted': predicted_y[0]}
            ), stack=False)

            # show button to download GMM model, whether trained or uploaded
            # read data as bytes then make st.download_button with specified file name
            gmm_bytesio = io.BytesIO()
            gmm.save_to_path(gmm_bytesio)

            gmm_bytes_data = gmm_bytesio.getvalue()
            st.download_button("Download GMM .pkl file", gmm_bytes_data, file_name=f'{gmm_name}.pkl')
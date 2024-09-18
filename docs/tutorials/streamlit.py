# Author: Sierra Janson

import streamlit as st
from UnsupervisedDeepTrainer import *
# use streamlit run streamlit.py to run

st.title('Brown Dwarf Spectra Analzyer Training Sandbox ✨')
st.write("To aid the accurate and swift determination of spectroscopic parameters from observational spectra.")
st.write('Train models, see results, and save models!')

# in depth settings or quick settings
# select standardization of data

st.sidebar.header("Model Hyperparameters for CNN")
epochs = st.sidebar.slider("Epochs", 1, 10, 3)
batch_size = st.sidebar.slider("Batch Size", 32, 256, 128)

model_type = st.selectbox(
    "Choose model",
    ("Autoencoder Feature Extractor CNN", "K-means", "DBScan")
)

use_pca = st.selectbox(
    "Use PCA on training dataset?",
    ("Yes", "No"),
)


if st.button("Train Model"):
    # make a load data and data processing section
    print("Begin training")

    # load data and extract features
    with st.spinner("Loading data..."):
        data_processor = load_data("TelescopeML_reference_data")
        data_processor = process_data(data_processor)

        # grab some flux values
        x_vals = np.array(data_processor.X_train_standardized_rowwise)

        # grab the atmospheric parameter columns
        ap    = data_processor.y_train_standardized_columnwise

        # create a dataframe out of the wavelength values 
        df = pd.DataFrame(x_vals)

        # initialize model
        model = UnsupervisedML(x_vals)

    # deploy chosen model
    if (model_type == "Supervised CNN"):
        pass

    elif (model_type == "Autoencoder Feature Extractor CNN"):
        history = None
        encoder = None
        print(batch_size,epochs)
        with st.spinner(f"Training model with {epochs} epochs and a batch size of {batch_size}..."):
            history, encoder = model.build_and_train_model(batch_size, epochs)
            print("Done with training...")
        fig, ax = plt.subplots()
        plt.plot(history.history['loss'])
        ax.set_title('Model Accuracy')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        st.pyplot(fig)


    elif (model_type == "K-means"):
        with st.spinner('Forming clusters...'):
            k_labels = model.kmeans()

        # if (st.button("Display Clusters")):
        #     xvar = st.selectbox("Choose a variable to plot clusters in respect to",
        #                             ("gravity", "c_o_ratio", "metallicity", "temperature"),)
        #     yvar = st.selectbox("Choose a variable to plot clusters in respect to",
        #                             ("gravity", "c_o_ratio", "metallicity", "temperature"),)
        #     if (st.button("Plot")):
        df['gravity'] = ap[:,[0]]
        df['c_o_ratio'] = ap[:,[1]]
        df['metallicity'] = ap[:,[2]]
        df['temperature'] = ap[:,[3]]

        kmeans_df = df
        kmeans_df['kmeans'] = k_labels
        fig, ax = plt.subplots()
        ax.set_title('Kmeans Clusters')
        xvar="c_o_ratio"
        yvar="temperature"
        ax.set_xlabel(xvar)
        ax.set_ylabel(yvar)

        ax.scatter(x=kmeans_df[xvar],y=kmeans_df[yvar],c=kmeans_df['kmeans'])
        ax.legend()
        st.pyplot(fig)


    elif (model_type == "DBScan"):
        with st.spinner('Forming clusters...'):
            db_labels = model.DBscan(.13)
        # if st.button("Display Clusters"):
        df['gravity'] = ap[:,[0]]
        df['c_o_ratio'] = ap[:,[1]]
        df['metallicity'] = ap[:,[2]]
        df['temperature'] = ap[:,[3]]

        kmeans_df = df
        kmeans_df['dbscan'] = db_labels
        fig, ax = plt.subplots()
        ax.set_title('DBscan Clusters')
        xvar="c_o_ratio"
        yvar="temperature"
        ax.scatter(x=kmeans_df[xvar],y=kmeans_df[yvar],c=kmeans_df['dbscan'])
        ax.legend()
        st.pyplot(fig)

    # model_buffer = io.BytesIO()
    # tf.keras.models.save_model(model, model_buffer, save_format='h5')
    # model_buffer.seek(0)

    # # Download button
    # st.download_button(
    #     label="Download Model",
    #     data=model_buffer,
    #     file_name='mnist_model.h5',
    #     mime='application/octet-stream'
    # )

footer="""
<style>
    .footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    text-align: center;
    }
</style>

<div class="footer">
    <p> Developed with ❤ by <a text-align: center;' href="https://github.com/sierrajanson" target="_blank">Sierra Janson</a> </p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
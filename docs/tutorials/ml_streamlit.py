import streamlit as st
from Unsupervised_utils import *
import pandas as pd
import os

st.title('TelescopeML Unsupervised Learning')
st.markdown('This is a web app for users to interact with different unsupervised learning algorithms. Data is the Brown Dwarf/Expolanet data provided in the TelescopeML project.  \n \n**Note**: Unsupervised methods work only with the features, thus handling of the labels is withheld.')
__reference_data_path__ = os.getenv("TelescopeML_reference_data")

train_BD = pd.read_csv(os.path.join(__reference_data_path__, 
                                    'training_datasets', 
                                    'browndwarf_R100_v4_newWL_v3.csv.bz2'), compression='bz2')
X = train_BD.drop(
    columns=['gravity', 
             'temperature', 
             'c_o_ratio', 
             'metallicity'])

y = train_BD[['gravity', 'c_o_ratio', 'metallicity', 'temperature', ]]
output_names = ['gravity', 'temperature', 'c_o_ratio', 'metallicity']

wavelength_names = [item for item in train_BD.columns.to_list() if item not in output_names]
wavelength_values = [float(item) for item in wavelength_names]

data_processor = Unsupervised_Data_Processor( 
                             flux_values=X.to_numpy(),
                             wavelength_names=X.columns,
                             wavelength_values=wavelength_values,
                             output_values=y.to_numpy(),
                             output_names=output_names,
                             spectral_resolution=200,
                                )

standardization = st.selectbox('Choose Standardization Method', ['Normalize X Column Wise', 'Normalize X Row Wise', 'Standardize X Column Wise', 'Standardize X Row Wise', 'None'], index = None, placeholder='Standardization Method...')


if standardization == "'Normalize X Column Wise":
    X = data_processor.normalize_X_column_wise(X=X)
elif standardization == "'Normalize X Row Wise":
    X = data_processor.normalize_X_row_wise(X=X)
elif standardization == "'Standardize X Column Wise":
    X = data_processor.standardize_X_column_wise(X=X)
elif standardization =='Standardize X Row Wise':
    X = data_processor.standardize_X_row_wise(X=X)
else: X=X    

    
method = st.selectbox('Choose Method', ['K-Means', 'Principal Component Analysis', 'DBSCAN'],index=None, placeholder="Select Unsupervised Method...")  

if method == 'K-Means':
    max_iter_select = st.number_input("Provide Max Iters", value=100, placeholder="Type a Number...")
    num_clusters_select = st.selectbox('Number of Clusters',[i for i in range(1,11)], index=None, placeholder="Select Number of Clusters...")
    feature1 = int(st.number_input('Select First Feature', value = 1, min_value=1, max_value=104, step=1))
    feature2 = int(st.number_input('Select Second Feature', value=2, min_value=1, max_value=104, step=1))

    if st.button("Run K-Means"):
        if num_clusters_select is not None:

            data = tf.convert_to_tensor(X, dtype=tf.float32)
            method = Unsupervised_Algorithms(data)
#             fig = method.kmeans(num_clusters=num_clusters_select, max_iter=max_iter_select)
            st.pyplot(method.kmeans(num_clusters=num_clusters_select, max_iter=max_iter_select, f1=feature1, f2=feature2))
            
if method == 'Principal Component Analysis':
    num_components_select = st.selectbox('Number of Components', [i for i in range(1, 11)], index = None, placeholder="Select Number of Components...")

    if st.button("Run PCA"):
        data = tf.convert_to_tensor(X, dtype=tf.float32)
        method = Unsupervised_Algorithms(data)
        st.pyplot(method.pca(num_components=num_components_select))

if method == "DBSCAN":
    eps_select = st.number_input("Provide Radius ",min_value=0.001, step=0.005,value=0.005,format="%f",placeholder="Type a Number...")
    min_val_select = st.number_input("Provide Minimum Number of Samples ", value=5, placeholder="Type a Number...")
    feature1 = int(st.number_input('Select First Feature', value = 1, min_value=1, max_value=104, step=1)) 
    feature2 = int(st.number_input('Select Second Feature', value=2, min_value=1, max_value=104, step=1))

    if feature1 == feature2:
        st.error("The two numbers cannot be the same. Please select different numbers.")
        
    if st.button("Run DBSCAN"):
#         feature1 = feature1-1
#         feature2 = feature2-1
        data = X.to_numpy()
        method = Unsupervised_Algorithms(data)
        st.pyplot(method.dbscan(eps=eps_select, min_sample=min_val_select, f1=feature1, f2=feature2))

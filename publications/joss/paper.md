---
title: '`TelescopeML` -- I. An End-to-End Python Package for Interpreting Telescope Datasets through Training 
Machine Learning Models, Generating Statistical Reports, and Visualizing Results'
tags:
  - Python
  - Astronomy
  - Exoplanets
  - Brown dwarfs
  - Spectroscopy
  - Atmospheric retrieval
  - Atmospheric models
  - Machine learning
  - Convolutional Neural Network
  - Telescope datasets
authors:
  - name: Ehsan (Sam) Gharib-Nezhad
    orcid: 0000-0002-4088-7262
    affiliation: "1, 2"
  - name: Natasha E. Batalha
    orcid: 0000-0003-1240-6844 
    affiliation: "1"
  - name: Hamed Valizadegan
    orcid: 0000-0001-6732-0840
    affiliation: "3, 4"
  - name: Miguel J. S. Martinho
    orcid: 0000-0002-2188-0807
    affiliation: "3, 4"
  - name: Mahdi Habibi
    affiliation: "5"
  - name: Gopal Nookula 
    affiliation: "6"
affiliations:
  - name: Space Science and Astrobiology Division, NASA Ames Research Center, Moffett Field, CA, 94035 USA
    index: 1
  - name: Bay Area Environmental Research Institute, NASA Research Park, Moffett Field, CA 94035, USA
    index: 2
  - name: Universities Space Research Association (USRA), Mountain View, CA 94043, USA
    index: 3
  - name: Intelligent Systems Division, NASA Ames Research Center, Moffett Field, CA 94035, USA
    index: 4
  - name: Institute for Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf, Dresden 01328, Germany
    index: 5
  - name: Department of Computer Science, University of California, Riverside, Riverside, CA 92507 USA
    index: 6

date: 17 November 2023
bibliography: paper.bib

aas-doi: LINK OF ApJ PAPER
aas-journal: The Astrophysical Journal 

--- 

# Statement of Need

We are on in a new era of space exploration, thanks to advancements in ground- and space-based 
telescopes, such as the James Webb Space Telescope [e.g., @JWST2023PASP] and CRIRES. These remarkable instruments collect high-resolution, high-signal-to-noise spectra 
from extrasolar planets [e.g., @Alderson2023Nature], and brown dwarfs [e.g., @Miles2023ApJ] atmospheres. Without an accurate interpretation of this data, the main objectives 
of space missions will not be fully accomplished. Different analytical and statistical methods, such as the chi-squared-test, 
Bayesian statistics as well as radiative-transfer atmospheric modeling packages have been developed 
[e.g., @batalha2019picaso; @MacDonald2023POSEIDON] to interpret the spectra. They utilize either forward- and/or retrieval-radiative transfer modeling to analyze the spectra and 
extract physical information, such as atmospheric temperature, metallicity, carbon-to-oxygen ratio, and surface gravity 
[@line2014systematic; @Iyer2023Sphinx; @Marley2015]. These atmospheric models rely on generating the physics and chemistry of these atmospheres for a wide range of thermal structures 
and compositions. In addition to Bayesian-based techniques, machine learning and deep learning methods have been developed in recent years 
for various astronomical problems, including confirming the classification of light curves for 
exoplanet validation [e.g., @Valizadegan2021], recognizing molecular features  [@Zingales2018ExoGAN] as well as interpreting brown dwarfs spectra using Random Forest technique 
[e.g., @Lueber2023RandomForesr_BDs]. Here, we present one of the first applications of deep learning and convolutional neural networks on the interpretation brown dwarf 
atmospheric datasets. The configuration of a CNN and the key concepts can be found in [@Goodfellow_2016DeepLearning; @KIRANYAZ2021].

With the continuous observation of these objects and the increasing amount of data, there is a 
critical need for a systematic pipeline to quickly explore the datasets and extract important physical from them. In the future we can expand our pipeline to exoplanet atmospheres, and use it to provide insights about the diversity of exoplanets and brown dwarfs' 
atmospheric compositions. Ultimately,  `TelescopeML` will help facilitate the long-term analysis of this data in research. `TelescopeML`
is an ML Python package with Sphinx-ed user-friendly documentation that provides both trained ML models and ML tools 
for interpreting observational data captured by telescopes.






# Functionality and Key Features
`TelescopeML` is a Python package comprising a series of modules, each equipped with specialized machine learning and 
statistical capabilities for conducting Convolutional Neural Networks (CNN) or Machine Learning (ML) training on datasets 
captured from the atmospheres of extrasolar planets and brown dwarfs. The tasks executed by the `TelescopeML` modules are 
outlined below and visualized in following Figure:

- **DataMaster module**: Performs various tasks to process the datasets, including:
    - Preparing inputs and outputs
    - Splitting the dataset into training, validation, and test sets
    - Scaling/normalizing the data
    - Visualizing the data
    - Conducting feature engineering
  
- **DeepTrainer module**: Utilizes different methods/packages such as TensorFlow to:
  - Build Convolutional Neural Networks (CNNs) model using the training examples
  - Utilize tuned hyperparameters
  - Fit/train the ML models
  - Visualize the loss and training history, as well as the trained model's performance
  
- **Predictor module**: Implements the following tasks to predict atmospheric parameters: 
  - Processes and predicts the observational datasets 
  - Deploys the trained ML/CNNs model to predict atmospheric parameters
  - Visualizes the processed observational dataset and the uncertainty in the predicted results
  
- **StatVisAnalyzer module**: Provides a set of functions to perform the following tasks: 
  - Explores and processes the synthetic datasets
  - Performs the chi-square test to evaluate the similarity between two datasets
  - Calculates confidence intervals and standard errors
  - Functions t



# Details on the synthetic dataset

The training dataset (or synthetic spectra) in this study is computed using the open-source atmospheric radiative 
transfer Python package, [`PICASO`](https://natashabatalha.github.io/picaso/) [e.g., @batalha2019picaso], based on the 
`Sonora-Bobcat` model grid generated for cloudless brown dwarf atmospheres by [@marley2021sonora]. This set encompasses 
30,888 synthetic spectra, each including 104 wavelengths (i.e., 0.897, 0.906, ..., 2.512 μm) and their corresponding flux 
values. Each of these spectra has four output variables attached to it: effective temperature, gravity, carbon-to-oxygen ratio, 
and metallicity. These synthetic spectra are utilized to interpret observational datasets and derive these four atmospheric parameters.
An example of the synthetic and observational dataset is shown in the following figure.

# Details on the CNN methodology for Multi-output Regression problem

Each row in the synthetic spectra has 104 input variables. The order of these data points and their magnitude are crucial 
to interpret the telescope data. For this purpose, we implemented a Convolutional Neural Network (CNN) method with 1-D convolutional 
layers. CNN is a powerful technique for this study because it extracts the dominant features from these spectra and then passes them 
to the fully connected hidden layers to learn the patterns. The output layer predicts the four atmospheric parameters.
An example of the CNN architecture is depicted in the following figure.




![TelescopeML main modules to manipulate the training example, build the ML model, train and tune it, and ultimately 
extract the target features from the observational data.](TelescopeML_Modules_Infograph.jpg){height="900pt"}


# Documentation


`TelescopeML` is available and being maintained as a GitHub repository at
[github.com/EhsanGharibNezhad/TelescopeML](https://github.com/EhsanGharibNezhad/TelescopeML). Online 
documentation is hosted with _Sphinx_ using _ReadtheDocs_ tools and includes several instructions and tutorials 
as follows: 

- **Main page**: [ehsangharibnezhad.github.io/TelescopeML/](https://ehsangharibnezhad.github.io/TelescopeML/)
- **Installation**: [ehsangharibnezhad.github.io/TelescopeML/installation.html](https://ehsangharibnezhad.github.io/TelescopeML/installation.html)
- **Tutorials and examples**: [ehsangharibnezhad.github.io/TelescopeML/tutorials.html](https://ehsangharibnezhad.github.io/TelescopeML/tutorials.html)
- **The code**: [ehsangharibnezhad.github.io/TelescopeML/code.html](https://ehsangharibnezhad.github.io/TelescopeML/code.html)
- **ML Concepts**: [ehsangharibnezhad.github.io/TelescopeML/knowledgebase.html](https://ehsangharibnezhad.github.io/TelescopeML/knowledgebase.html)


#  Users  and Future Developments

Astrophysicists with no prior machine learning knowledge can deploy the `TelescopeML` package and download the 
pre-trained ML or CNN models to interpret their observational data. In this scenario, pre-trained ML models, 
as well as the PyPI package, can be installed and deployed following the online instructions. Tutorials in the 
Sphinx documentation include examples for testing the code and also serve as a starting point. For this purpose, 
a basic knowledge of Python programming is required to install the code, run the tutorials, deploy the modules, 
and extract astronomical features from the datasets. The necessary machine learning background and a detailed 
guide for package installation, along with links to further Python details, are provided to help understand 
the steps and outputs.

Astrophysicists with machine learning expertise and data scientists can also benefit from this package by 
developing and fine-tuning the modules and pre-trained models to accommodate more complex datasets from 
various telescopes. This effort could also involve the utilization of new ML and deep learning algorithms, 
adding new capabilities such as feature engineering methods, and further optimization of hyperparameters 
using different and more efficient statistical techniques. The ultimate outcome from these two groups would 
be the creation of more advanced models with higher performance and robustness, as well as the extension of 
the package to apply to a wider range of telescope datasets.


# Utilized Underlying Packages
For processing datasets and training ML models in `TelescopeML`, the following software/packages are employed:
Scikit-learn [@scikit-learn], TensorFlow [@tensorflow2015-whitepaper], AstroPy [@astropy:2022], SpectRes [@SpectRes],
Pandas [@reback2020pandas], NumPy [@harris2020array], SciPy [@2020SciPy-NMeth], Matplotlib [@Hunter:2007], 
Seaborn [@Waskom2021], Bokeh [@bokeh]. Additionally, for generating training astronomical datasets, 
Picaso [@batalha2019picaso] is implemented.


# Acknowledgements
EGN and GN would like to thank OSTEM internships and funding through the NASA with contract number 80NSSC22DA010.
EGN acknowledges ChatGPT 3.5 for proofreading some of the functions and providing helpful suggestions. 

# References

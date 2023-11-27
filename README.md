# TelescopeML

[![PyPI - Latest Release](https://img.shields.io/pypi/v/TelescopeML.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.python.org/pypi/TelescopeML)
[![PyPI - Python Versions](https://img.shields.io/pypi/pyversions/TelescopeML.svg?logo=python&logoColor=white&label=Python)](https://pypi.python.org/pypi/TelescopeML)
[![Build Status](https://app.travis-ci.com/EhsanGharibNezhad/TelescopeML.svg?branch=main)](https://app.travis-ci.com/EhsanGharibNezhad/TelescopeML)

``TelescopeML`` is a Python package comprising a series of modules, each equipped with specialized machine learning and
statistical capabilities for conducting Convolutional Neural Networks (CNN) or Machine Learning (ML) training on
datasets captured from the atmospheres of extrasolar planets and brown dwarfs. The tasks executed by the ``TelescopeML``
modules are outlined below:


![](docs/figures/TelescopeML_modules.png)

- *DataMaster module*: Performs various tasks to process the datasets, including:

    - Preparing inputs and outputs
    - Splitting the dataset into training, validation, and test sets
    - Scaling/normalizing the data
    - Visualizing the data
    - Conducting feature engineering

- *DeepTrainer module*: Utilizes different methods/packages such as TensorFlow to:

  - Build Convolutional Neural Networks (CNNs) model using the training examples
  - Utilize tuned hyperparameters
  - Fit/train the ML models
  - Visualize the loss and training history, as well as the trained model's performance

- *Predictor module*: Implements the following tasks to predict atmospheric parameters:

  - Processes and predicts the observational datasets
  - Deploys the trained ML/CNNs model to predict atmospheric parameters
  - Visualizes the processed observational dataset and the uncertainty in the predicted results

- *StatVisAnalyzer module*: Provides a set of functions to perform the following tasks:

  - Explores and processes the synthetic datasets
  - Performs the chi-square test to evaluate the similarity between two datasets
  - Calculates confidence intervals and standard errors
  - Functions to visualize the datasets, including scatter plots, histograms, boxplots


or simply...

 - Load the trained CNN models
 - Follow the tutorials
 - Predict the stellar/exoplanetary parameters
 - Report the statistical analysis



## Documentation

- Documentation: https://ehsangharibnezhad.github.io/TelescopeML/
- Installation: https://ehsangharibnezhad.github.io/TelescopeML/installation.html
- Tutorials: https://ehsangharibnezhad.github.io/TelescopeML/tutorials.html
- The code: https://ehsangharibnezhad.github.io/TelescopeML/code.html


.. TelescopeML documentation master file, created by
   sphinx-quickstart on Tue Dec 27 15:39:09 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TelescopeML
==============


``TelescopeML`` is a Python package comprising a series of modules, each equipped with specialized machine learning and
statistical capabilities for conducting Convolutional Neural Networks (CNN) or Machine Learning (ML) training on
datasets captured from the atmospheres of extrasolar planets and brown dwarfs. The tasks executed by the ``TelescopeML``
modules are outlined below:

    - *StatVisAnalyzer*: Explore and process the synthetic datasets (or the training examples) and perform statistical analysis.
    - *DeepBuilder*: Specify training and target features, normalize/scale datasets, and construct a CNN model.
    - *DeepTrainer*: Create an ML model, train the model with the training examples, and utilize hyperparameters.
    - *Predictor*: Train the module using specified hyperparameters.

or simply...

    - Load the pre-trained CNN models based on the latest synthetic datasets
    - Predict the stellar/(exo-)planetary parameters
    - Report the statistical analysis

.. image:: figures/TelescopeML_modules.png
  :width: 1100





======================

.. toctree::
   :maxdepth: 1
   :hidden:
  
   Installation <installation>
   Tutorials <tutorials>
   The Code <code>
   KnowledgeBase <knowledgebase> 
   Github <https://github.com/ehsangharibnezhad>
   Publications <publications>
   What to Cite <cite>



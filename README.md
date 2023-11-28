# TelescopeML

[![PyPI - Latest Release](https://img.shields.io/pypi/v/TelescopeML.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.python.org/pypi/TelescopeML)
[![Build Status](https://app.travis-ci.com/EhsanGharibNezhad/TelescopeML.svg?branch=main)](https://app.travis-ci.com/EhsanGharibNezhad/TelescopeML)
[![.github/workflows/draft-pdf.yml](https://github.com/EhsanGharibNezhad/TelescopeML/actions/workflows/draft-pdf.yml/badge.svg)](https://github.com/EhsanGharibNezhad/TelescopeML/actions/workflows/draft-pdf.yml)
[![pages-build-deployment](https://github.com/EhsanGharibNezhad/TelescopeML/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/EhsanGharibNezhad/TelescopeML/actions/workflows/pages/pages-build-deployment)

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


## Contributors

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->


<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="20%">
        <a href="https://github.com/EhsanGharibNezhad">
          <img src="https://avatars.githubusercontent.com/u/22139918?v=4?s=100" width="100px;" alt="Ehsan Gharib-Nezhad"/><br />
          <sub><b>Ehsan Gharib-Nezhad</b></sub>
        </a><br/>
        <a href="https://github.com/EhsanGharibNezhad/TelescopeML/commits?author=EhsanGharibNezhad" title="Code">💻</a> 
        <a href="#ideas" title="Ideas, Leading ">🤔</a>
        <a href="https://pypi.org/project/TelescopeML/" title="Maintenance">🚧</a>
        <a href="https://ehsangharibnezhad.github.io/TelescopeML/tutorials.html" title="tutorial">📚</a>
      </td>
      <td align="center" valign="top" width="20%">
        <a href="http://natashabatalha.github.io">
          <img src="https://avatars.githubusercontent.com/u/6554465?v=4?s=100" width="100px;" alt="Natasha Batalha"/><br />
          <sub><b>Natasha Batalha</b></sub>
        </a><br/>
        <a href="#mentoring-astro" title="mentoring">🧑‍🏫</a> 
        <a href="https://github.com/EhsanGharibNezhad/TelescopeML/commits?author=natashabatalha" title="bug">🐛</a>
        <a href="#ideas" title="Ideas & Feedback">🤔</a>
      </td>
      <td align="center" valign="top" width="20%">
        <a href="https://github.com/hvalizad">
          <img src="https://avatars.githubusercontent.com/u/52180694?v=4?s=100" width="100px;" alt="Hamed Valizadegan"/><br />
          <sub><b>Hamed Valizadegan</b></sub>
        </a><br/>
        <a href="#mentoring-ML" title="mentoring">🧑‍🏫</a> 
        <a href="#ideas" title="Ideas & Feedback">🤔</a>
      </td>
      <td align="center" valign="top" width="20%">
        <a href="https://github.com/migmartinho">
          <img src="https://avatars.githubusercontent.com/u/47117139?v=4?s=100" width="100px;" alt="Miguel Martinho"/><br />
          <sub><b>Miguel Martinho</b></sub>
        </a><br/>
        <a href="" title="mentoring-CNNTuning-BOHB" title="Mentoring">🧑‍🏫</a>
        <a href="#ideas" title="Ideas & Feedback">🤔</a>
      </td>
      <td align="center" valign="top" width="20%">
        <a href="https://github.com/letgotopal">
          <img src="https://avatars.githubusercontent.com/u/89670109?v=4?s=100" width="100px;" alt="Gopal Nookula"/><br />
          <sub><b>Gopal Nookula</b></sub>
        </a><br/>
        <a href="https://ehsangharibnezhad.github.io/TelescopeML/tutorials.html" title="tutorial">📚</a>
      </td>
    </tr>
  </tbody>
</table>



<!-- ALL-CONTRIBUTORS-LIST:END -->
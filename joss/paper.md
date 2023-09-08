---
title: '`TelescopeML`: Deep Convolutional Neural Networks and Machine Learning Models for Analyzing Stellar and Exoplanetary Telescope Spectra'
tags:
  - Python
  - astronomy
  - exoplanets
  - brown dwarfs
  - spectroscopy
  - atmospheric models
authors:
  - name: Ehsan (Sam) Gharib-Nezhad
    orcid: 0000-0002-4088-7262
    affiliation: "1, 2"
affiliations:
  - name: Space Science and Astrobiology Division, NASA Ames Research Center, Moffett Field, CA, 94035 USA
    index: 1
  - name: Bay Area Environmental Research Institute, CA, USA
    index: 2

date: 28 Augest 2023
bibliography: paper.bib

[//]: # (aas-doi: LINK OF APJS PAPER)
[//]: # (aas-journal: The Astrophysical Journal Supplement Series)

--- 

# Summary
One of NASA's primary objectives is to fully characterize extrasolar atmospheres and understand the chemical and physical 
processes that govern their formation and evolution. An accurate interpretation of these objects requires full analysis of 
the observational datasets recorded by telescopes to extract their atmospheric parameters such as
surface temperatures, gravity, and metallicity. The analysis of these datasets involves building atmospheric 
models (e.g., forward and retrival) as well as thorough statistical analysis of the results from these models [e.g., @Marley2015]. 

`TelescopeML` is a Python Package which provides an end-to-end pipline for astrophysicists to apply the trained Convolutional Neural Networks (CNNs) models 
to the telescope datasets to extract critical atmospheric parameters. `TelescopeML` has these following main modules: 
1) **DeepBuilder** that is 
responsible to process the telescope synthetic datasets (e.g., normalizing, feature engeenering, train, val, test set splitting and defining 
the feature targets and training examples).
2) **DeepTrainer** which it takes the prepared synthetic dataset and defined hyperparameters to train the CNNs model.
3) **Predictor** load the observational telescope dataset as well as the trained CNNs models to extract of the atmospheric parameters.
4) **StatVisAnalyzer** provides a set of statistical, visualizing, and printing functions to analyze the input and output datasets, and create insights toward the ML results.
This module is utilized by all other modules during the entire Build-Train-Predict process.



# Statement of Need


# Future Developments

`POSEIDON` v1.0 officially supports the modelling and retrieval of exoplanet transmission spectra in 1D, 2D, and 3D. The initial release also includes a beta version of thermal 
emission spectra modelling and retrieval (for cloud-free, 1D atmospheres, with no scattering), which will be developed further in future releases. Suggestions for additional 
features are more than welcome.

# Documentation

Documentation for `TelescopeML`, with step-by-step tutorials illustrating research applications, is available at 
[https://poseidon-retrievals.readthedocs.io/en/latest/](https://poseidon-retrievals.readthedocs.io/en/latest/). 

# Similar Tools

The following exoplanet retrieval codes are open source: [`PLATON`](https://github.com/ideasrule/platon) [@Zhang:2019; @Zhang:2020], 
[`petitRADTRANS`](https://gitlab.com/mauricemolli/petitRADTRANS) 

# Acknowledgements

[//]: # (EGN expresses gratitude to the developers of many open source Python packages used by `TelescopeML`, in particular `Numba` [@Lam:2015], `numpy` [@Harris:2020], `Matplotlib` )
[//]: # ([@Hunter:2007], `SciPy` [@Virtanen:2020], and `Spectres` [@Carnall:2017].)

[//]: # (EGN acknowledges financial support )

# References

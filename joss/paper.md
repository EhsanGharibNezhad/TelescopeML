---
title: '`telescopeML`: '
tags:
  - Python
  - astronomy
  - exoplanets
  - spectroscopy
  - atmospheric retrieval
  - atmospheric models
  - JWST
authors:
  - name: Ehsan (Sam) Gharib-Nezhad
    orcid: 0000-0002-4088-7262
    affiliation: "1, 2"
affiliations:
  - name: Space Science and Astrobiology Division, NASA Ames Research Center, Moffett Field, CA, USA
    index: 1
  - name: Bay Area Environmental Research Institute, CA, USA
    index: 2

date: 04 Augest 2023
bibliography: paper.bib

aas-doi: LINK OF APJS PAPER
aas-journal: The Astrophysical Journal Supplement Series

--- 

# Summary

![Schematic architecture of the `POSEIDON` atmospheric retrieval code. Users can call `POSEIDON` in two main ways: (i) to generate a model exoplanet spectrum for a specified planet 
atmosphere (green arrows); or (ii) to fit an observed exoplanet spectrum by statistical sampling of a model's atmospheric properties (purple arrows). The diagram highlights code 
inputs (circles), algorithm steps (rectangles), and code outputs (bottom green or purple boxes). \label{fig:POSEIDON_architecture}](figures/POSEIDON_Architecture_2022){width=100%}

`POSEIDON` was first described in the exoplanet literature by [@MacDonald:2017]. Since then, the code has been used in 17 peer-reviewed publications [e.g., @Alam:2021; 
@Sedaghati:2017; @Kaltenegger:2020]. Most recently, a detailed description of `POSEIDON`'s new multidimensional forward model, `TRIDENT`, was provided by [@MacDonald:2022].

# Statement of Need


# Future Developments

`POSEIDON` v1.0 officially supports the modelling and retrieval of exoplanet transmission spectra in 1D, 2D, and 3D. The initial release also includes a beta version of thermal 
emission spectra modelling and retrieval (for cloud-free, 1D atmospheres, with no scattering), which will be developed further in future releases. Suggestions for additional 
features are more than welcome.

# Documentation

Documentation for `POSEIDON`, with step-by-step tutorials illustrating research applications, is available at 
[https://poseidon-retrievals.readthedocs.io/en/latest/](https://poseidon-retrievals.readthedocs.io/en/latest/). 

# Similar Tools

The following exoplanet retrieval codes are open source: [`PLATON`](https://github.com/ideasrule/platon) [@Zhang:2019; @Zhang:2020], 
[`petitRADTRANS`](https://gitlab.com/mauricemolli/petitRADTRANS) [@Molliere:2019], [`CHIMERA`](https://github.com/mrline/CHIMERA) [@Line:2013], 
[`TauRex`](https://github.com/ucl-exoplanets/TauREx3_public) [@Waldmann:2015; @Al-Refaie:2021], [`NEMESIS`](https://github.com/nemesiscode/radtrancode) [@Irwin:2008] [`Pyrat 
Bay`](https://github.com/pcubillos/pyratbay) [@Cubillos:2021], and [`BART`](https://github.com/exosports/BART) [@Harrington:2022]

# Acknowledgements

RJM expresses gratitude to the developers of many open source Python packages used by `POSEIDON`, in particular `Numba` [@Lam:2015], `numpy` [@Harris:2020], `Matplotlib` 
[@Hunter:2007], `SciPy` [@Virtanen:2020], and `Spectres` [@Carnall:2017].

RJM acknowledges financial support from the UK's Science and Technology Facilities Council (STFC) during the early development of `POSEIDON` and support from NASA Grant 
80NSSC20K0586 issued through the James Webb Space Telescope Guaranteed Time Observer Program. Most recently, RJM acknowledges support from NASA through the NASA Hubble Fellowship 
grant HST-HF2-51513.001 awarded by the Space Telescope Science Institute, which is operated by the Association of Universities for Research in Astronomy, Inc., for NASA, under 
contract NAS5-26555. RJM is especially grateful to Lorenzo Mugnai and Michael Zhang for excellent and helpful referee reports, and to the editor, Dan Foreman-Mackey, for his 
tireless efforts to encourage new people to join the open source community in astronomy. RJM thanks Nikole Lewis, Ishan Mishra, Jonathan Gomez Barrientos, John Kappelmeier, Antonia 
Peters, Kath Landgren, and Ruizhe Wang for helpful discussions.

# References

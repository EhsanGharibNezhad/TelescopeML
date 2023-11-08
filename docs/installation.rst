Installation
=============

.. toctree::
   :maxdepth: 1
   :caption: Contents:



.. note::
	`TelescopeML` requires python >= 3.8.


Step 1: Create your directory structure
----------------------------------------
Let’s start by creating the folder structure. Here is the folder structure of
an example project, named TelescopeML_project with two sub-directories: *reference_data* and
*notebooks*.


| TelescopeML_project
| ├── reference_data
| │   ├── training_datasets
| │   ├── observational_datasets
| │   ├── figures
| │   └── trained_ML_models
| └── notebooks
|

.. note::
    You need to be inside the `TelescopeML_project` directory if you want to follow method 2
    (i.e., `Install with Git`) and clone the github repository.


Step 2: Download Input Data and Trained ML Models
--------------------------------------------------
First things first, after creating the home directory, you need  to download the following datasets, 1--4,
as well as the trained CNN models to fully utilize the code and apply it to your datasets.


    1.  for **training_datasets/** directory: `Link to the pre-trained ML models to deploy <https://zenodo.org/records/2459971/files/EGNMRL__H2O__1000K__1E+00bar__H2He.XS.bz2?download=1>`_
    2.  for **observational_datasets/** directory:  `Link the synthetic Brown-dwarf training datasets <https://stackoverflow.com/>`_
    3.  for **trained_ML_models/** directory:  `Link to the observational telescope spectra for few brown dwarfs <https://stackoverflow.com/>`_
    4.  for **notebooks/** directory: `Download the Jupyter notebook tutorials to deploy the ML models <https://stackoverflow.com/>`_



Step 3: Install the Package
----------------------------

Method 1: Install via Pip (Straightforward)
+++++++++++++++++++++++++++++++++++++++++++
The easiest way to install the most stable version is with *pip*, the Python package manager,
but do not forget that you still need to create a virtual environment using the `Anaconda distribution <https://www.anaconda.com/download/>`_
and then install ``TelescopeML`` there by the following steps:

1. Create a conda virtual environment using `python>=3.8`:

.. code-block:: bash

    conda create --name  TelescopeML  python=3.9

2. Activate the new environment:

.. code-block:: bash

    conda activate TelescopeML

3. Now install `TelescopeML` via pip:

.. code-block:: bash

    pip install TelescopeML



Method 2: Install with Git (Recommended)
+++++++++++++++++++++++++++++++++++++++++++
If you want to access the latest features or modify the code and contribute, we suggest that you clone the source code
from GitHub by following steps:

1. Clone the repo and Create `Conda` environment named *TelescopeML*:

.. code-block:: bash

    git clone https://github.com/ehsangharibnezhad/TelescopeML.git
    cd TelescopeML
    conda env create -f environment.yml



2. Activate the new environment:

.. code-block:: bash

    conda activate TelescopeML

3. Install the library via the `setup.py` file:

.. code-block:: bash

    python3 setup.py develop


Now, you should have the package installed alongside the trained models and telescope datasets.

Step 4: Set input file environment variables
---------------------------------------------


For Mac OS
++++++++++++++++++++++++++++++++++++++++++++

follow the following steps to set the link to the input data:

1. Check your default shell in your terminal:

.. code-block:: bash

    echo $SHELL

This command will display the path to your default shell, typically something
like `/bin/bash` or `/bin/zsh`, or `/bin/sh`.

2. Set the environment variables :

    * If your shell is `/bin/zsh`:

    .. code-block:: bash

        echo 'export $TelescopeML_reference_data="/PATH_TO_YOUR_TelescopeML_project/" ' >>~/.zshrc
        source ~/.zshrc
        echo $TelescopeML_reference_data


    * if your shell is `/bin/bash`:

    .. code-block:: bash

        echo 'export $TelescopeML_reference_data="/PATH_TO_YOUR_TelescopeML_project/"' >>~/.bash_profile
        source ~/.bash_profile
        echo $TelescopeML_reference_data

    * if your sell is `/bin/sh`:

    .. code-block:: bash

        echo 'export $TelescopeML_reference_data="/PATH_TO_YOUR_TelescopeML_project/"' >>~/.profile
        source ~/.profile
        echo $TelescopeML_reference_data


.. note::
    - Replace `PATH_TO_YOUR_TelescopeML_project` with the actual path to your TelescopeML directory.
    - *echo* command is used to check that your variable has been defined properly.


For Linux
++++++++++
In Linux, the choice between `~/.bashrc` and `~/.bash_profile` depends on your specific use case and how
you want environment variables to be set, but `~/.bashrc` is a common and practical choice for
modern Linux system.

.. code-block:: bash

    echo 'export $TelescopeML_reference_data="/PATH_TO_YOUR_TelescopeML_project/" ' >>~/.bashrc
    source ~/.bashrc
    echo $TelescopeML_reference_data


Replace `PATH_TO_YOUR_TelescopeML_project` with the actual path to your TelescopeML directory.

For Windows
++++++++++++

For Windows users, we recommend installing Windows Subsystem for Linux (WSL) before proceeding.
WSL is a compatibility layer for Windows that allows you to run a Linux distribution alongside
your Windows installation.




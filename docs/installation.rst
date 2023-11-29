Installation
=============

.. toctree::
   :maxdepth: 1
   :caption: Contents:



.. note::
    `TelescopeML` requires python >= 3.9.


Step 1: Create your directory structure
----------------------------------------
Let’s start by creating the folder structure as follow, named *TelescopeML_project*. While you are inside this parent
*TelescopeML_project* directory, download the *reference_data* folder which include the following sub-directories:
*training_datasets*, *tutorials*, *observational_datasets*,  *figures*, *trained_ML_models*.

Download link for **reference_data** folder is: `Link <https://zenodo.org/doi/10.5281/zenodo.10183098>`_

| TelescopeML_project
| ├── reference_data
| │   ├── training_datasets
| │   ├── tutorials
| │   ├── observational_datasets
| │   ├── figures
| │   └── trained_ML_models
|





Step 2: Set input file environment variables
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

        echo 'export TelescopeML_reference_data="/PATH_TO_YOUR_reference_data/" ' >>~/.zshrc
        source ~/.zshrc
        echo $TelescopeML_reference_data


    * if your shell is `/bin/bash`:

    .. code-block:: bash

        echo 'export TelescopeML_reference_data="/PATH_TO_YOUR_reference_data/"' >>~/.bash_profile
        source ~/.bash_profile
        echo $TelescopeML_reference_data

    * if your sell is `/bin/sh`:

    .. code-block:: bash

        echo 'export TelescopeML_reference_data="/PATH_TO_YOUR_reference_data/"' >>~/.profile
        source ~/.profile
        echo $TelescopeML_reference_data


.. note::
    - Replace `PATH_TO_YOUR_reference_data` with the actual path to your *reference_data* folder
      that you downloaded in step 1.
    - *echo* command is used to check that your variable has been defined properly.


For Linux
++++++++++
In Linux, the choice between `~/.bashrc` and `~/.bash_profile` depends on your specific use case and how
you want environment variables to be set, but `~/.bashrc` is a common and practical choice for
modern Linux system.

.. code-block:: bash

    echo 'export TelescopeML_reference_data="/PATH_TO_YOUR_TelescopeML_project/" ' >>~/.bashrc
    source ~/.bashrc
    echo $TelescopeML_reference_data



Step 3: Install the Package
----------------------------

.. note::
    You need to first have `Anaconda distribution <https://www.anaconda.com/download/>`_ installed on your machine,
    before proceed to the next steps.

Method 1: Install through Git (Recommended)
+++++++++++++++++++++++++++++++++++++++++++
If you want to access the latest features or modify the code and contribute, we suggest that you clone the source code
from GitHub by following steps:

.. note::
    For best practise, it is recommended to be inside the `TelescopeML_project` parent directory and then
    clone the github repository.

1. Clone the repo and Create `Conda` environment named *TelescopeML*:

.. code-block:: bash

    git clone https://github.com/ehsangharibnezhad/TelescopeML.git
    cd TelescopeML
    conda env create -f environment.yml



2. Activate the new environment:

.. code-block:: bash

    conda activate TelescopeML

3. Install the library via the `setup.py` file inside *TelescopeML* directory:

.. code-block:: bash

    python3.9 setup.py develop


Now, you should have the latest version of the package installed alongside the reference data.


4. Test the package by going to the *docs/tutorials/* directory and run all notebooks there using *jupyter-lab*.

Method 2: Install though Pip (Straightforward)
++++++++++++++++++++++++++++++++++++++++++++++
The easiest way to install the most stable version is with *pip*, the Python package manager,
but do not forget that you still need to create a virtual environment using the `Anaconda distribution <https://www.anaconda.com/download/>`_
and then install ``TelescopeML`` there by the following steps:

1. Create a conda virtual environment using `python>=3.8`:

.. code-block:: bash

    conda create --name  TelescopeML  python=3.9

2. Activate the new environment:

.. code-block:: bash

    conda activate TelescopeML

3. Now install the latest PyPI version of `TelescopeML` via pip:

.. code-block:: bash

    pip install TelescopeML


4. Test the package by going into the *reference_data/tutorials/* directory and run all notebooks there using *jupyter-lab*.


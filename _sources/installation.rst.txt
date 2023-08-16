Installation
=============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. note::

    ``TelescopeML`` is written in `python3` and therefore requires an environment with python>=3.8.
    To utilize all modules in the library in full capacity (e.g., explore data, train CNNs, predict results),
    the following methods are recommended:


Method 1: Install with Git (Recommended)
-----------------------------------------
If you want to access the latest features or modify the code and contribute,
we suggest that you clone the source code from GitHub by following steps:

1. Clone the repo and create Anaconda environment named `TelescopeML`:

.. code-block:: bash

    git clone https://github.com/ehsangharibnezhad/TelescopeML.git
    cd TelescopeML
    conda TelescopeML create -f environment.yml


2. Activate the new environment and verify it:

.. code-block:: bash

    conda activate TelescopeML
    conda env list

3. Install the library via the `setup.py` file:

.. code-block:: bash

    git clone https://github.com/ehsangharibnezhad/TelescopeML.git
    cd TelescopeML
    python3 setup.py install

If you are thinking to develop the code, then you can install it using:

.. code-block:: bash

    git clone https://github.com/ehsangharibnezhad/TelescopeML.git
    cd TelescopeML
    python3 setup.py develop

Method 2: Install via Pip
-----------------------------

You can install `TelescopeML` package from pip, the Python package manager, by the following steps:

1. Clone the repo and Create Anaconda environment named :

.. code-block:: bash

    git clone https://github.com/ehsangharibnezhad/TelescopeML.git
    cd TelescopeML
    conda TelescopeML create -f environment.yml


2. Activate the new environment and verify it:

.. code-block:: bash

    conda activate TelescopeML
    conda env list

3. Now install via pip:

.. code-block:: bash

    pip install TelescopeML
    
Read more details at `Conda Tutorials
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file>`_


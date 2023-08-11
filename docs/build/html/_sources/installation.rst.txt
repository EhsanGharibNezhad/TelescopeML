Installation
=============

Requirements
~~~~~~~~~~~~

`TelescopeML` is written in python3 and therefore requires an environment with python>=3.8.

Method 1: Install with Git for *users*:
----------------------------------------

The Github repository contains the reference folder and helpful tutorials.  

.. code-block:: bash 

	git clone https://github.com/ehsangharibnezhad/TelescopeML.git
	cd TelescopeML
	python setup.py install 




Create Conda Environment: 
--------------------------

Method 1: Creating an environment from the **environment.yml** file
...................................................................

1. Create the environment named `telescopeML` from the `environment.yml` file:

.. code-block:: bash

	conda telescopeML create -f environment.yml


2. Activate the new environment: 

.. code-block:: bash

    conda activate telescopeML


3. Verify that the new environment was installed correctly:

.. code-block:: bash

    conda env list
    
Get more details at `Conda Tutorials 
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file>`_

from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()



setup(
    name='TelescopeML',
    version = (this_directory / "version.txt").read_text(),  # MAJOR.MINOR.PATCH
    description = 'An End-to-End Python Package for Interpreting Telescope Datasets through Training Machine Learning Models, Generating Statistical Reports, and Visualizing Results',
    long_description = long_description,
    long_description_content_type='text/markdown',
    author = 'Ehsan (Sam) Gharib-Nezhad',
    author_email = 'e.gharibnezhad@gmail.com',
    url = 'https://ehsangharibnezhad.github.io/TelescopeML',
    license = 'GPL-3.0',
    download_url = 'https://github.com/EhsanGharibNezhad/TelescopeML',
    classifiers = [
                  'Intended Audience :: Science/Research',
                  'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                  'Operating System :: OS Independent' ,
                  'Programming Language :: Python',
                  'Programming Language :: Python :: 3',
                  'Topic :: Scientific/Engineering :: Astronomy',
                  'Topic :: Software Development :: Libraries :: Python Modules'
  ],
  packages=find_packages(exclude=('tests', 'docs')),
  install_requires=[
                  'numpy==1.26.1',
                  'bokeh',
                  'pandas',
                  'astropy',
                  'matplotlib',
                  'seaborn==0.12.2',
                  'sphinx==7.2.6',
                  'scipy==1.11.1',
                  'keras==2.14.0',
                  'tensorflow==2.14.0',
                  'jupyterlab',
                  'sphinx',
                  'spectres==2.2.0',
                  'scikit-learn==1.3.0',
  ],
    zip_safe = False,
)

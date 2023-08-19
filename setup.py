
# try:
#     from setuptools import setup
# except ImportError:
#     from ez_setup import use_setuptools
#     use_setuptools()
#     from setuptools import setup

from setuptools import setup, find_packages

setup(
    name='TelescopeML',
    version = '0.0.0',
    description = 'Deep Convolutional Neural Networks and Machine Learning Models for Analyzing Stellar and Exoplanetary Telescope Spectra',
    long_description = 'README.md',
    author = 'Ehsan (Sam) Gharib-Nezhad',
    author_email = 'e.gharibnezhad@gmail.com',
    url = 'https://ehsangharibnezhad.github.io/TelescopeML',
    license = 'GPL-3.0',
    download_url = 'https://github.com/EhsanGharibNezhad/TelescopeML',
    classifiers = [
                  'Intended Audience :: Science/Research',
                  'License :: GNU General Public License v3.0',
                  'Operating System :: OS Independent' ,
                  'Programming Language :: Python',
                  'Programming Language :: Python :: 3',
                  'Topic :: Scientific/Engineering :: Computer Science, Astronomy',
                  'Topic :: Software Development :: Libraries :: Python Modules'
  ],
  packages=find_packages(exclude=('tests', 'docs')),
  install_requires=[
          'numpy',
          'bokeh',
          # 'holoviews',
          'pandas',
          # 'joblib',
          # 'photutils',
          'astropy',
          'matplotlib',
          # 'pysynphot',
          'sphinx',
          'scipy',
          ],
    zip_safe = False,
)
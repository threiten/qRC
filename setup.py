from setuptools import setup

# Check if ROOT is installed and raise error if not found
try:
    import ROOT
except ModuleNotFoundError:
    raise ImportError('ROOT not found. You need a full working installation of ROOT to install this package.\n' \
            'For more info, see: https://root.cern/install/')

# Check if yaml is installed
# The inclusion as 'install_requires' is not straightforward, take care of this at a later time
try:
    import yaml
except ModuleNotFoundError:
    raise ModuleNotFoundError('Yaml is needed and cannot be installed automatically. Please install it and try again.')

setup(
    name = "quantile_regression_chain",
    author = "Thomas Reitenspiess",
    packages = ['quantile_regression_chain', 'quantile_regression_chain.tmva'],
    install_requires = [
        'numpy',
        'sklearn',
        'pandas',
        'uproot4',
        'xgboost',
        'joblib',
        ]
)

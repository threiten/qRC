# Quantile Regression Chain for Data/MC Corrections

## Motivation
TODO

## Installation
After cloning the repository and entering it, run:
```bash
$ python setup.py install
```
to install the package.
All the dependencies should be handled by the ```setup.py``` script itself.

## Run
The process is rather long and involves submission of jobs on a cluster running [SLURM](https://slurm.schedmd.com/) and internal parallelization that can be performed either with [Ray](https://ray.io/) or [IPyParallel](https://ipyparallel.readthedocs.io/en/latest/#). 
The ```utils``` directory contains directories numbered from 1 to 9 with the code and instructions to run, plus directory (```setup_ray```) with the scripts to setup a Ray cluster for the internal parallelization needed in some steps.

Following are the main steps:

1. TagAndProbe NTuples production
2.  Pandas dataframes production
3. Data training
4. Montecarlo training
5. Bayesian optimization
6. Final training and IdMVA computation
7. Plot
8. Systematic uncertainties 
9. Regressors conversion

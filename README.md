## Motivation
TODO

## Installation
The suggested way to run this software is in a conda environment with the necessary prerequisites (```environment.yaml``` to set up a conda environment will be provided).
At this stage, the workflow consists simply in cloning the repository and running 
```bash
python setup.py install
```
to install in the default directories, allowing to import the package from everywhere.

## Workflow

### Create dataframes
(did not change)

### Train regressors
Given a simple script ```train_regressors.py``` like the following:
```python
import argparse
from dask.distributed import Client, LocalCluster, progress, wait, get_client  
  
from quantile_regression_chain import QRCScheduler  
  
import logging  
logger = logging.getLogger("")  
  
def parse_arguments():  
    parser = argparse.ArgumentParser(  
        description = 'Variables related to local user paths')  
  
    parser.add_argument(  
        "-c",  
        "--config_file",  
        required=True,  
        type=str,  
        help="Path to YAML config file")  
  
    parser.add_argument(  
        "-sc",  
        "--slurm_config",  
        type=str,  
        help="Path to YAML config file with information to setup a SLURM cluster")  
  
    return parser.parse_args()  
  
def setup_logging(output_file, level=logging.DEBUG):  
    logger.setLevel(level)  
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")  
  
    handler = logging.StreamHandler()  
    handler.setFormatter(formatter)  
    logger.addHandler(handler)  
  
    file_handler = logging.FileHandler(output_file, "w")  
    file_handler.setFormatter(formatter)  
    logger.addHandler(file_handler)  
  
def main(args):  
    # Parse cmd line arguments  
    config_file = args.config_file  
    slurm_config = args.slurm_config
  
    qrc_scheduler = QRCScheduler(config_file)  
  
    if slurm_config:  
        qrc_scheduler.setup_slurm_cluster(slurm_config)  
  
    qrc_scheduler.load_dataframes()  
  
    qrc_scheduler.train_regressors()  
  
if __name__ == "__main__":  
    args = parse_arguments()  
    setup_logging('train_all_with_scheduler.log', logging.INFO)    
    main(args)
```
the regressors for all the variables from UL2017 can be trained by running:
```bash
python train_regressors.py --config_file config.yaml
```
where ```config.yaml``` looks like this:
```python
dataframes:
  data:  
    EB:  
      SS:  
        df_data_EB_train.h5  
      iso:  
        df_data_EB_Iso_train.h5  
    EE:  
      SS:  
        df_data_EE_train.h5  
      iso:  
        df_data_EE_Iso_train.h5  
  mc:  
    EB:  
      SS:  
        df_mc_EB_train.h5  
      iso:  
        df_mc_EB_Iso_train.h5  
    EE:  
      SS:  
        df_mc_EE_train.h5  
      iso:  
        df_mc_EE_Iso_train.h5  
  
year:  
  2017  
  
datasets:  
  ['data', 'mc']  
  
detectors:  
  ['EB', 'EE']  
  
variables:  
  SS:  
    ['probeCovarianceIeIp', 'probeS4', 'probeR9', 'probePhiWidth', 'probeSigmaIeIe', 'probeEtaWidth']  
  iso:  
  ch:  
    ['probeChIso03', 'probeChIso03worst']  
  ph:  
    ['probePhoIso']  
  
quantiles:  
  [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
  
work_dir:  
  /path/to/dataframes  
  
weights_dir:  
  /path/to/weights_dir
```
It is also possible to run on a SLURM cluster with the following command
```bash
python train_all.py --config_file config.yaml --slurm_config slurm_config.yaml
```
where ```slurm_config.yaml``` is a file that looks like this
```python
jobqueue:  
  slurm:  
    cores: 1  
    memory: 10GB  
    jobs: 4
```

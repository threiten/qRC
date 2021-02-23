# Train MC Shapes

Train 508 regressors.

## Shower Shapes

For each of the 6 variables the 21 quantiles are trained in parallel. The output of the 21 quantiles for the *n*-th variable becomes the input for the 21 quantiles for variable *n*-th + 1.
At the level of quantiles, the parallelization is performed is done internally using either Ray or IPyParallel.
Assuming we want to train for the barrel using 4700000 events using Ray as a backend (with a cluster running at the address ```192.33.123.23:6379```) we run the following command:
```bash
$ python train_qRC_MC.py --EBEE EB --config config/config_qRC_training_5M.yaml --n_evts 4700000 --backend Ray --clusterid 192.33.123.23:6379
```

## Isolations

Commands to run:
```bash
$ python train_qRC_I_MC.py --EBEE EB --config config/config_qRC_training_ChI_5M.yaml --n_evts 4700000 --backend Ray --clusterid 192.33.123.23:6379
$ python train_qRC_I_MC.py --EBEE EE --config config/config_qRC_training_ChI_5M.yaml --n_evts 4700000 --backend Ray --clusterid 192.33.123.23:6379
$ python train_qRC_I_MC.py --EBEE EB --config config/config_qRC_training_PhI_5M.yaml --n_evts 4700000 --backend Ray --clusterid 192.33.123.23:6379
$ python train_qRC_I_MC.py --EBEE EE --config config/config_qRC_training_PhI_5M.yaml --n_evts 4700000 --backend Ray --clusterid 192.33.123.23:6379
```

## Check outputs

Once both data and MC regressors we can check if there is anything missing in the ```weightsDir``` by running:
```bash
$ python check_output_names.py -d path_to_weightsDir
```

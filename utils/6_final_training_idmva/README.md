# Final Training & IdMVA

MC regressors are here used to train a lower number of regressors (the ones that, after being converted to ```xml``` format, will be uploaded to flashgg).

First of all, the results from the previous step (bayesian optimization) have to be copied in ```config/finalRegression_settings.yaml```.

The following commands
```bash
$ python train_final_Reg_SS.py --EBEE EB --config config/config_qRC_training_5M.yaml --n_evts 4700000 --backend Ray --clusterid 192.33.123.23:6379
$ python train_final_Reg_SS.py --EBEE EE --config config/config_qRC_training_5M.yaml --n_evts 4700000 --backend Ray --clusterid 192.33.123.23:6379
$ python train_final_Reg_Iso.py --EBEE EB --config config/config_qRC_training_ChI_5M.yaml --n_evts 4700000 --backend Ray --clusterid 192.33.123.23:6379
$ python train_final_Reg_Iso.py --EBEE EE --config config/config_qRC_training_ChI_5M.yaml --n_evts 4700000 --backend Ray --clusterid 192.33.123.23:6379
$ python train_final_Reg_Iso.py --EBEE EB --config config/config_qRC_training_PhI_5M.yaml --n_evts 4700000 --backend Ray --clusterid 192.33.123.23:6379
$ python train_final_Reg_Iso.py --EBEE EE --config config/config_qRC_training_PhI_5M.yaml --n_evts 4700000 --backend Ray --clusterid 192.33.123.23:6379
```
produce the final regressors in the directory specified in the config files by ```outDir```.

The last step consists in running:
```bash
$ python make_corr_df.py --config config/config_make_corr_df.yaml --EBEE EB -N 4700000 --backend Ray --clusterid 192.33.123.23:6379  --final --mvas  
$ python make_corr_df.py --config config/config_make_corr_df.yaml --EBEE EE -N 4700000 --backend Ray --clusterid 192.33.123.23:6379 --final --mvas
```
(maybe also specifying ```--n_jobs```) which produces in ```workDir``` the following files:
```
df_data_EB_test_IdMVA_5M.h5
df_data_EE_test_IdMVA_5M.h5
df_mc_EB_test_corr_clf_5M.h5
df_mc_EE_test_corr_clf_5M.h5
```

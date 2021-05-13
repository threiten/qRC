# Systematic Uncertainties

The idea in this part is to derive two new sets of corrections, each trained with half the original sample. In simpler words, we perform everything we have done so far once again on two smaller datasets.

## 1. Split dataframes
The relevant files can be found inside ```1_split_dataframes```.

To split the dataframes, run
```bash
$ python split_dataframes.py --n_evts 2000000 --input_dir workDir
```
where ```workDir``` is the absolute path to where the original pandas dataframes are located.

After running, the following new files should be present at that location:
```bash
df_data_EB_Iso_train_spl1.h5 
df_data_EE_Iso_train_spl1.h5 
df_mc_EB_Iso_train_spl1.h5 
df_mc_EE_Iso_train_spl1.h5   
df_data_EB_Iso_train_spl2.h5 
df_data_EE_Iso_train_spl2.h5 
df_mc_EB_Iso_train_spl2.h5 
df_mc_EE_Iso_train_spl2.h5  
df_data_EB_train_spl1.h5 
df_data_EE_train_spl1.h5 
df_mc_EB_train_spl1.h5 
df_mc_EE_train_spl1.h5   
df_data_EB_train_spl2.h5 
df_data_EE_train_spl2.h5 
df_mc_EB_train_spl2.h5 
df_mc_EE_train_spl2.h5 
```

## 2. Train Data

Go back to ```utils/3_data_training```.
On a cluster that uses SLURM, it should be enough to run:
```bash
./train_all_systematics.sh
```

## 3. Train MC

Go back to ```utils/4_mc_training```.
Set up a Ray cluster like explained in the above mentioned directory and run the following:
```bash
$ python train_qRC_MC.py --EBEE EB --config config/config_qRC_training_5M.yaml --n_evts 1000000 --backend Ray --clusterid 192.33.123.23:6379 -s 1
$ python train_qRC_MC.py --EBEE EE --config config/config_qRC_training_5M.yaml --n_evts 1000000 --backend Ray --clusterid 192.33.123.23:6379 -s 1
$ python train_qRC_MC.py --EBEE EB --config config/config_qRC_training_5M.yaml --n_evts 1000000 --backend Ray --clusterid 192.33.123.23:6379 -s 2
$ python train_qRC_MC.py --EBEE EE --config config/config_qRC_training_5M.yaml --n_evts 1000000 --backend Ray --clusterid 192.33.123.23:6379 -s 2

$ python train_qRC_I_MC.py --EBEE EB --config config/config_qRC_training_ChI_5M.yaml --n_evts 1000000 --backend Ray --clusterid 192.33.123.23:6379 -s 1
$ python train_qRC_I_MC.py --EBEE EE --config config/config_qRC_training_ChI_5M.yaml --n_evts 1000000 --backend Ray --clusterid 192.33.123.23:6379 -s 1
$ python train_qRC_I_MC.py --EBEE EB --config config/config_qRC_training_PhI_5M.yaml --n_evts 1000000 --backend Ray --clusterid 192.33.123.23:6379 -s 1
$ python train_qRC_I_MC.py --EBEE EE --config config/config_qRC_training_PhI_5M.yaml --n_evts 1000000 --backend Ray --clusterid 192.33.123.23:6379 -s 1
$ python train_qRC_I_MC.py --EBEE EB --config config/config_qRC_training_ChI_5M.yaml --n_evts 1000000 --backend Ray --clusterid 192.33.123.23:6379 -s 2
$ python train_qRC_I_MC.py --EBEE EE --config config/config_qRC_training_ChI_5M.yaml --n_evts 1000000 --backend Ray --clusterid 192.33.123.23:6379 -s 2
$ python train_qRC_I_MC.py --EBEE EB --config config/config_qRC_training_PhI_5M.yaml --n_evts 1000000 --backend Ray --clusterid 192.33.123.23:6379 -s 2
$ python train_qRC_I_MC.py --EBEE EE --config config/config_qRC_training_PhI_5M.yaml --n_evts 1000000 --backend Ray --clusterid 192.33.123.23:6379 -s 2
```

## 4. IdMVA

Training the final regressors is not needed for this stage, but we need the IdMVA. 
Move back to ```utils/6_final_training_idmva```.

```bash
$ python make_corr_df.py --config config/config_make_corr_df_spl1.yaml --EBEE EB -N 4700000 --backend Ray --clusterid 192.33.123.23:6379 --mvas  
$ python make_corr_df.py --config config/config_make_corr_df_spl1.yaml --EBEE EE -N 4700000 --backend Ray --clusterid 192.33.123.23:6379 --mvas  
  
$ python make_corr_df.py --config config/config_make_corr_df_spl2.yaml --EBEE EB -N 4700000 --backend Ray --clusterid 192.33.123.23:6379 --mvas  
$ python make_corr_df.py --config config/config_make_corr_df_spl2.yaml --EBEE EE -N 4700000 --backend Ray --clusterid 192.33.123.23:6379 --mvas
```
Consider specifying the flag ```--n_jobs```.
The names of the outputs can be seen in the config files.

## 5. Plot Ratios

To be sure that everything is working as expected, let's plot the ratios plots for the two sets of regressors. 
To do so, go back to ```utils/7_plots```.

```bash
$ python run_plotter.py --mc path_to/df_mc_EB_test_corr_clf_5M_spl1.h5 --data path_to/df_data_EB_test_IdMVA_5M_spl1.h5 --con  
fig config/config_EB_nofinals.yaml --outdir EB_s1_output_dir --norm --ratio -k  
$ python run_plotter.py --mc path_to/df_mc_EB_test_corr_clf_5M_spl2.h5 --data path_to/df_data_EB_test_IdMVA_5M_spl2.h5 --conf  
ig config/config_EB_nofinals.yaml --outdir EB_s2_output_dir --norm --ratio -k
$ python run_plotter.py --mc path_to/df_mc_EE_test_corr_clf_5M_spl1.h5 --data path_to/df_data_EE_test_IdMVA_5M_spl1.h5 --con  
fig config/config_EE_nofinals.yaml --outdir EE_s1_output_dir --norm --ratio -k  
$ python run_plotter.py --mc path_to/df_mc_EE_test_corr_clf_5M_spl2.h5 --data path_to/df_data_EE_test_IdMVA_5M_spl2.h5 --conf  
ig config/config_EE_nofinals.yaml --outdir EE_s2_output_dir --norm --ratio -k
```

## 6. Plot bands + syst shift graphs

This part is supposed to produce the plots showing the systematic uncertainties bands for IdMVA and a root file containing a graph with the amount of shift to be added for a certain IdMVA value.
Move to ```utils/7_plots```.
The script to use is ```produce_systematics.py```.
Run the following (both for EB and EE):
```bash
$ python produce_systematics.py -i path_to/df_mc_EE_test_corr_clf_5M.h5 \  
-s path_to/df_mc_EE_test_corr_clf_5M_spl1.h5 \  
-t path_to/df_mc_EE_test_corr_clf_5M_spl2.h5 \  
-d path_to/df_data_EE_test_IdMVA_5M.h5 \  
--factor 2. \  
--shiftF para \  
--outfile path_to/SystematicsIDMVA_LegRunII_v1_UL2018.root \  
--plotDir output_dir \  
--forceReweight
```

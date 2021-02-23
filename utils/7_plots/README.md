# Plots

Now it's time to take a look at the results and check if the corrections we trained perform their job correctly.
Three types of plots have to be performed in this part:

- ratios
- profiles
- correlation matrices

## Ratios

The script to run is ```run_plotter.py```, the relevant config files are ```config/config_EB.yaml``` and ```config/config_EE.yaml```.

Run the following:
```bash
$ python run_plotter.py --mc path_to/df_mc_EB_test_corr_clf_5M.h5 --data path_to/df_data_EB_test_IdMVA_5M.h5 --config config/config_EB.yaml --outdir path_to_output_dir --norm --ratio -k
$ python run_plotter.py --mc path_to/df_mc_EE_test_corr_clf_5M.h5 --data path_to/df_data_EE_test_IdMVA_5M.h5 --config config/config_EE.yaml --outdir path_to_output_dir --norm --ratio -k
```

## Profiles

The script to run is ```run_profile_plotter.py```

```bash
$ python run_profile_plotter.py --mc path_to/df_mc_EB_test_corr_clf_5M.h5 --data path_to/df_data_EB_test_IdMVA_5M.h5 --ou  
tdir path_to_output_dir --EBEE EB --varrs SS --final_reg  
$ python run_profile_plotter.py --mc path_to/df_mc_EB_test_corr_clf_5M.h5 --data path_to/df_data_EB_test_IdMVA_5M.h5 --ou  
tdir path_to_output_dir --EBEE EB --varrs Iso --final_reg
$ python run_profile_plotter.py --mc path_to/df_mc_EE_test_corr_clf_5M.h5 --data path_to/df_data_EE_test_IdMVA_5M.h5 --ou  
tdir path_to_output_dir --EBEE EE --varrs SS --final_reg  
$ python run_profile_plotter.py --mc path_to/df_mc_EE_test_corr_clf_5M.h5 --data path_to/df_data_EE_test_IdMVA_5M.h5 --ou  
tdir path_to_output_dir --EBEE EE --varrs Iso --final_reg
```

## Correlation Matrices

The script to run is ```run_corr_matrices.py```

```bash
$ python run_corr_matrices.py --mc path_to/df_mc_EB_test_corr_clf_5M.h5 --data path_to/df_data_EB_test_IdMVA_5M.h5 --outd  
ir path_to_output_dir --EBEE EB
$ python run_corr_matrices.py --mc path_to/df_mc_EB_test_corr_clf_5M.h5 --data path_to/df_data_EB_test_IdMVA_5M.h5 --outd  
ir path_to_output_dir --EBEE EE
```

# Make Dataframes

Create input dataframes for the training.
In the following:
```bash
python make_dataframes.py \  
-D /work/gallim/root_files/tnp_merged_outputs/2018/UNCORRECTED_20201231 \  
-O /work/gallim/tmp/tnp \  
-y 2018 \  
-E EE \  
-s 0.5
```
we have the following arguments:

- ```-D``` specifies the directory where the files ```outputData.root``` and ```outputMC.root``` are expected to be found;
- ```-O``` specifies the output directory for the pandas dataframes
- ```-s``` amount of train vs test data (0.5 means the dataset is split in half)

Running the above command twice (once with ```-E EB``` and once with ```-E EE```) will produce the following dataframes, which are all the ones we need:
```
df_data_EB_Iso_test.h5 
df_data_EB_test.h5 
df_data_EE_Iso_test.h5 
df_data_EE_test.h5 
df_mc_EB_Iso_test.h5 
df_mc_EB_test.h5 
df_mc_EE_Iso_test.h5 
df_mc_EE_test.h5  
df_data_EB_Iso_train.h5 
df_data_EB_train.h5 
df_data_EE_Iso_train.h5 
df_data_EE_train.h5 
df_mc_EB_Iso_train.h5 
df_mc_EB_train.h5 
df_mc_EE_Iso_train.h5 
df_mc_EE_train.h5
```

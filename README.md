# Chained quantile regression
This repo contains the code to do data/MC correction using chained quantile regression and stochastic matching. 
The class `quantileRegression_chain` can be used to correct a set of continious variables differentially and 
while keeping their correlations. The class `quantileRegression_chain_disc` can be used to correct discontinious variables.
## Training BDTs for quantiles
To train the BDTs that will be used to extract the conditional pdf the functions `trainOnData` for data and `trainOnMC` 
for MC have to be used. For example:
```python
import quantileRegression_chain as qRegC
qRC = qRegC.quantileRegression_chain(year,EBEE,workDir,variables)
qRC.loadDataDF(df_name,0,stop,rsh,columns)
qRC.trainOnData(variable,weightsDir)
```
# Scripts for training
The strategy to train on a large dataset is the following
1. Train on data

	To train on data use `scripts/run_qRC_training.sh`
	```bash
	./run_qRC_training.sh <config_file_ShowerShapes>.yaml <config_file_PhotonIso>.yaml <config_file_ChargedIsos>.yaml <n_evts> <EB/EE>
	```
	This will submit one job per quantile per variable to the SGE queue via qub. BEWARE: There is a hard coded path in this script. Change it accordingly
	
2. Train Shower Shapes on MC
   
   To train the shower shape correction for MC use `training/train_qRC_MC.py`
   ```bash
   python train_qRC_MC.py  -c <config_file_ShowerShapes>.yaml -N <n_evts> -E <EB/EE> -B <cluster_profile> -i <cluster_id>
   ```
   
3. Train Isolations on MC

	To train the shower shape correction for MC use `training/train_qRC_MC.py`
   ```bash
   python train_qRC_I_MC.py  -c <config_file_(PhotonIso/ChargedIsos)>.yaml -N <n_evts> -E <EB/EE> -B <cluster_profile> -i <cluster_id>
   ```

## Final corrections training

After validating the initial training, one can train the final single regressors that can be used to apply the corrections to the simulation in production. To do so, follow these steps:

1. Train the final shower shape corrections

	To train the final shower shape correction use `training/train_final_Reg_SS.py`
	```bash
	python train_final_Reg_SS.py  -c <config_file_ShowerShapes>.yaml -N <n_evts> -E <EB/EE> -B <ipython_cluster_profile> -i <cluster_id> -n 21
	```
2. Train final charged Iso corrections

	To train the final correction for the charged isolations use `training/train_final_Reg_Iso.py`
	```bash
	python train_final_Reg_Iso.py  -c <config_file_(ChargedIsos)>.yaml -N <n_evts> -E <EB/EE> -B <ipython_cluster_profile> -i <cluster_id> -n 21
	```
	
3. Train final photon Iso corrections

	To train the final correction for the photon isolation use `training/train_final_Reg_Iso.py`
	```bash
	python train_final_Reg_Iso.py  -c <config_file_(PhotonIso)>.yaml -N <n_evts> -E <EB/EE> -B <ipython_cluster_profile> -i <cluster_id> -n 21
	```
	
	The only difference between the command for charged and photon Iso are the config files
	
### Note on config files

In general the config files for the training for data and simulation for the initial and final training have the same format. Examples can be found in `examples` 

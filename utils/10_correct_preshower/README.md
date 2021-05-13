# Correct Preshower

For what concerns samples from 2018, the corrections don't seem to perform extremely well for EE. This can be seen both in RR (slightly) and UL (more). The reason for this could be that the radiation damage in the ECAL preshower has become big enough to be noticed in this year and, since we don't take this into account in the MC samples, we have to also correct the variable ```esEnergyOverSCRawEnergy```.

The strategy consists in training the usual 21 bdts + finals for the new variable, taking care of the following: 

- the **corrected** shower shapes have to become new input features of the bdts
- when producing the training (and test) dataframes, apply the cut ```abs(probeScEta)>1.653```

Following is the complete procedure to produce the updated results.

### 1. Dump new Flashgg ntuples

The variable we want to correct has to be added to the list of the dumped variables in flashgg. To do this, [this branch](https://github.com/maxgalli/flashgg/tree/UseScEnergyForIdMVA) was used. 
N.B.: the two new variables dumped where stupidly called ```phoIdMVA_ESEffSigmaRR``` and ```phoIdMVA_esEnovSCRawEn```; this called problems in the qRC code because in some parts it relies on the variables not having any ```_``` in the name; next time, just call them ```ESEffSigmaRR``` and ```esEnergyOverSCRawEnergy``` respectively.

### 2. Re-run the whole process
Basically everything has to be re-done:

1. **Create Pandas dataframes**: go back to ```2_pandas_dfs_production``` and remember to add the cut ```abs(probeScEta)>1.653``` to the queries that are present in ```make_dataframes.py```. After this, dig inside the code and check if the function ```loadROOT``` fetches all the columns of the TTree when creating the pandas dataframe; if not change it, since we will need both the corrected and uncorrected values form the TTree.
2. **Train data**: same idea as before, go back to ```3_data_training```; the new files are 
		-	```train_preshower.sh```
		-	```SLURM/run_data_training_preshower.sh```
		-	```SLURM/job_train_qRC_preshower_data.sh```
		-	```python_scripts/train_qRC_preshower_data.py```
		-	```config/config_qRC_training_preshower.yaml```
	running
	```
	$ ./train_preshower.sh
	```
	should be enough.
	Note that these files follow the same scheme of the ones used before, with the difference that the attribute ```kinrho```, which contains the 4 variables used to feed the first node of the chain in the previous part, is extended with the names of the shower shapes variables, which are intended to be the corrected ones (since, as already said, the *un*corrected ones have the suffix ```_uncorr```).
3. **Train MC**: follow the instructions reported in ```4_mc_training```, using the new files called ```train_qRC_preshower_MC.py``` and ```config/config_qRC_training_preshower.yml```.
Also in this case, the input features to the regressors consist in the 4 kinematic variables + the 6 corrected shower shapes.
4. **Final Training**: here we train the "final regressors" and produce the dataframes from which we will get the quantities to plot. Go back to  ```6_final_training_idmva```. Run
	```
	python train_final_Reg_preshower.py --EBEE EE --config config/config_qRC_training_preshower.yaml --n_evts 4700000
	```
	to train the regressors and 
	```
	python make_corr_preshower_df.py --config config/config_make_corr_df_preshower.yaml --EBEE EE -N 4700000 --final
	``` 
	to produce the dataframe.
5. **PhoId "on the fly" and plots**:  in this case, the PhotonID is recomputed on the fly when plotting. Remember to change the following:
		- in ```quantile_regression_chain/tmva/IdMVAComputer.py``` we need to change the way the PhotonID is computed, meaning that it has to take as input the corrected version of the preshower variable; to do so, apply the changes listed in [this commit](https://github.com/maxgalli/qRC/commit/a66b1475e19d5b86c18a172c909b5409d45424bd); in the list ```columns```, the name ```probeEnovSCRawEn``` has to be changed to the name of the preshower variable that was dumped from flashgg; also the variable ```probePhoIso03``` has to be changed to ```probePhoIso```, since apparently the name is different from flashgg to the correction framework;
		- for the plotting part, go back to ```7_plots```; here you can find the horribly written ```run_plotter_preshower.py``` and ```config/config_EE_preshower.yaml```; for what concerns ```run_plotter_preshower.py``` the idea is to plot the values for the PhotonID for data + MC corrected by flashgg + MC SS corrected preshower uncorrected recomputed on the fly + MC SS corrected preshower corrected recomputed on the fly; an example can be seen [here](https://gallim.web.cern.ch/gallim/plots/Hgg/QRC/Legacy2018/Training/Nominals/EE_etacut/ratios/dataMC_probePhoIdMVA_0.png). Since some quantities necessary for the recomputation might not be present in the final dataframes, the originally created ones are also accessed (```original_data``` and ```original_mc``` in the code). Whoever will run this the next time will probably have to change it quite a bit.
	To run it, something like 
	```
	python run_plotter_preshower.py   --mc /work/gallim/dataframes/UL2018/correct_preshower/full_dataframes_etacut/df_mc_EE_test_corr_clf_5M.h5   --data /work/gallim/dataframes/UL2018/correct_preshower/full_dataframes_etacut/df_data_EE_test.h5  --config my_config_ES_EE_test.yaml   --outdir /eos/home-g/gallim/www/plots/Hgg/QRC/Legacy2018/Training/Nominals/EE_etacut/ratios --norm --ratio -k   --recomp_mva  --reweight_cut "abs(probeScEta)>1.56 and tagPt>40 and probePt>25 and mass>80 and mass<100 and probePassEleVeto==0 and abs(tagScEta)<2.5 and abs(probeScEta)<2.5"
	```
	should work.
	
6. **Systematics**: in general, the same instructions reported in ```8_systematic_uncertainties``` have to be followed, with some changes:
	
	- move to ```2_pandas_dfs_production``` and use the split option to produce splitted train dataframes 
	-  in ```3_data_training```, the new files to use are ```train_preshower_systematics.sh```, ```SLURM/run_data_training_preshower_systematics.sh``` and ```SLURM/job_train_qRC_preshower_data_systematics.sh```
	- go back to ```4_mc_training``` and run ```
python train_qRC_I_preshower_MC.py --EBEE EE --config config/config_qRC_training_preshower.yml --n_evts 1000000 --backend ray --clusterid 192.33.123.23:6379 -s 1``` and  
```python train_qRC_I_preshower_MC.py --EBEE EE --config config/config_qRC_training_preshower.yml --n_evts 1000000 --backend ray --clusterid 192.33.123.23:6379 -s 2```
	- the PhoID computation is trickier: we can't use the SS+Iso corrected inside flashgg, but we should instead use the regressors we trained previously; everything is taken care of inside ```6_final_idmva/make_corr_preshower_df_split.py```. Remember that before doing this the names of the shower shapes found in the ```X``` value of the dictionary saved inside the pkl files have to be added the suffix ```_corr```; either this, or doing it the other way around inside ```make_corr_preshower_df_split.py```
	- go back to ```7_plots``` and run
```
python produce_systematics_preshower.py -i /work/gallim/root_files/tnp_merged_outputs/2018/Preshower/outputMC.root \  
-s /work/gallim/dataframes/UL2018/correct_preshower/full_dataframes_etacut/df_mc_EE_test_corr_clf_5M_spl1.h5 \  
-t /work/gallim/dataframes/UL2018/correct_preshower/full_dataframes_etacut/df_mc_EE_test_corr_clf_5M_spl2.h5 \  
-d /work/gallim/root_files/tnp_merged_outputs/2018/Preshower/outputData.root \  
--mc_tree tagAndProbeDumper/trees/DYJetsToLL_amcatnloFXFX_13TeV_All \  
--data_tree tagAndProbeDumper/trees/Data_13TeV_All \  
--factor 2. \  
--shiftF para \  
--plotDir /eos/home-g/gallim/www/plots/Hgg/QRC/Legacy2018/FlashggOutput/Systematics/bands/EE_CorrPreshower_etacut/uncorr_edges \  
--cut "abs(probeScEta)>1.56 and tagPt>40 and probePt>25 and mass>80 and mass<100 and probePassEleVeto==0 and abs(tagScEta)<2.5 and abs(probeScEta)<2.5"  
  
python produce_systematics_preshower.py -i /work/gallim/root_files/tnp_merged_outputs/2018/Preshower/outputMC.root \  
-s /work/gallim/dataframes/UL2018/correct_preshower/full_dataframes_etacut/df_mc_EE_test_corr_clf_5M_spl1.h5 \  
-t /work/gallim/dataframes/UL2018/correct_preshower/full_dataframes_etacut/df_mc_EE_test_corr_clf_5M_spl2.h5 \  
-d /work/gallim/root_files/tnp_merged_outputs/2018/Preshower/outputData.root \  
--mc_tree tagAndProbeDumper/trees/DYJetsToLL_amcatnloFXFX_13TeV_All \  
--data_tree tagAndProbeDumper/trees/Data_13TeV_All \  
--factor 2. \  
--shiftF para \  
--outfile /work/gallim/dataframes/UL2018/correct_preshower/full_dataframes_etacut/SystematicsIDMVA_LegRunII_v1_UL2018.root \  
--plotDir /eos/home-g/gallim/www/plots/Hgg/QRC/Legacy2018/FlashggOutput/Systematics/bands/EE_CorrPreshower_etacut/corr_edges \  
--correctEdges \
--cut "abs(probeScEta)>1.56 and tagPt>40 and probePt>25 and mass>80 and mass<100 and probePassEleVeto==0 and abs(tagScEta)<2.5 and abs(probeScEta)<2.5"
``` 

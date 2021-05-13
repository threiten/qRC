# Bayesian Optimization

Here we run a bayesian optimization to get the parameters to use in the final regression.
For performance reasons, this part is run only on gpus. 
On PSI Tier3, the way to submit jobs to the gpu queue consists in:

- submit a fake job (shell, sleep or whatever) on the gpu queue;
- once this is done, you can ```ssh``` to the node containing the gpus and run the commands manually 

N.B.: for this a different environment with xgboost compiled specifically for GPU execution is needed. 

Once on the node with gpus, run the following:
```bash
$ for var in "probeCovarianceIeIp" "probeR9" "probeS4" "probeSigmaIeIe" "probeEtaWidth" "probePhiWidth" "probeChIso03" "probeChIso03worst" "probePhoIso"; do sbatch SLURM/run_bayOpt_gpu.sub ${var};  done
```
(absolute paths might have to be added).

This produces a text file per variable in the ```outputs``` directory.

At this point, the best iteration for each variable has to be chosen by checking the best tradeoff between score and overtraining. 
From inside the ```outputs``` directory, run:
```bash
$ python select_parameters_directory.py --directory .
```
to get a fair suggestion.

#!/bin/bash

export PATH=/work/mdonega/anaconda3/bin:$PATH

source activate xgboost

for var in "probeCovarianceIeIp"; # "probeR9" "probeS4" "probeSigmaIeIe" "probeEtaWidth" "probePhiWidth" "probeChIso03" "probeChIso03worst" "probePhoIso";
do
    sbatch /t3home/threiten/python/qRC/bayOpt/gpu/run_bayOpt_gpu.sh ${var}
done

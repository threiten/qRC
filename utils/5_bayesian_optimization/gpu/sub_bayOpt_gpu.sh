#!/bin/bash

#for var in "probeCovarianceIeIp" "probeR9" "probeS4" "probeSigmaIeIe" "probeEtaWidth" "probePhiWidth" "probeChIso03" "probeChIso03worst" "probePhoIso";
for var in "probeS4" "probeEtaWidth" "probePhiWidth";
do
    sbatch /t3home/gallim/devel/qRC/utils/bayOpt/gpu/run_bayOpt_gpu.sh ${var}
done

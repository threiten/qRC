#!/bin/bash

configSS=$1
configPI=$2
configCI=$3
N_evts=$4
EBEE=$5


cd /t3home/threiten/python/qRC/scripts/

for var in "probeCovarianceIetaIphi" "probeSigmaIeIe" "probeEtaWidth" "probePhiWidth" "probeR9" "probeS4";
do
    echo Submitting training jobs for variable $var on data
    for q in 0.01 0.99 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95;
    do
	echo Submitting training job for quantile $q on data
	qsub -q all.q -l h_vmem=6G job_train_qRC_data.sh ${configSS} ${var} ${q} ${N_evts} ${EBEE}
    done
done

echo Training jobs for shower shapes submitted!

for var in "probePhoIso";
do
    echo Submitting training jobs for variable $var on data
    for q in 0.01 0.99 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95;
    do
	echo Submitting training job for quantile $q on data
	qsub -q all.q -l h_vmem=6G job_train_qRC_I_data.sh ${configPI} ${var} ${q} ${N_evts} ${EBEE}
    done
done

echo Training jobs for Photon Iso submitted!

for var in "probeChIso03" "probeChIso03worst";
do
    echo Submitting training jobs for variable $var on data
    for q in 0.01 0.99 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95;
    do
	echo Submitting training job for quantile $q on data
	qsub -q all.q -l h_vmem=6G job_train_qRC_I_data.sh ${configCI} ${var} ${q} ${N_evts} ${EBEE}
    done
done

echo Training jobs for Charged Iso submitted!

cd -

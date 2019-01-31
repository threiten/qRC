#!/bin/bash

# i=0
workDir=$1
dataFrame_data=$2
weightsDir=$3
year=$4
N_evts=$5
EBEE=$6


# while [ $(qstat -q all.q -s r | wc -l) -le $((N_jobs+2)) ]
# do 
#     sleep 10
# done

# echo All ipengine jobs have started, start training!

# ipcluster start --profile=long_6gb -n ${N_jobs} &

# sleep 1m

# while [ $i -lt ${N_jobs} ]
# do
# for var in "probeSigmaIeIe" "probeEtaWidth" "probePhiWidth" "probeR9" "probeS4" "probeCovarianceIeIp";
# do
#     for q in 0.01 0.99 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95;
#     do
# 	echo Submitting training jobs for quantile $q on data
# 	qsub -q short.q -l h_vmem=6G job_train_qRC_data.sh ${var} ${q} ${workDir} ${dataFrame_data} ${weightsDir} ${year} ${N_evts}
# 	# -l h_vmem=6G
#     # i=$(($i+6))
#     # echo Submitting training jobs for quantile $q on MC
#     # qsub -q long.q -l h_vmem=6G job_train_qRC_MC.sh $q ${workDir} ${dataFrame_MC} ${weightsDir} $i ${year} ${N_evts}
#     # i=$(($i+6))
#     done
# done

for var in "probeChIso03"; #"probeSigmaIeIe" "probeEtaWidth" "probePhiWidth" "probeR9" "probeS4";
do
    for q in 0.01 0.99 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95;
    do
	echo Submitting training jobs for quantile $q on data
	qsub -q short.q -l h_vmem=6G job_train_qRC_I_data.sh ${var} ${q} ${workDir} ${dataFrame_data} ${weightsDir} ${year} ${N_evts} ${EBEE}
	# -l h_vmem=6G
    # i=$(($i+6))
    # echo Submitting training jobs for quantile $q on MC
    # qsub -q long.q -l h_vmem=6G job_train_qRC_MC.sh $q ${workDir} ${dataFrame_MC} ${weightsDir} $i ${year} ${N_evts}
    # i=$(($i+6))
    done
done
# done
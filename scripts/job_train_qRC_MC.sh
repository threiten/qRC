#!/bin/bash

echo SWITCH OFF DISPLAY
export DISPLAY=

source /mnt/t3nfs01/data01/shome/threiten/jupyter_env.sh

quantile=$1
workDir=$2
dataFrame=$3
weightsDir=$4
indWorker=$5
year=$6
N_evts=$7

python /mnt/t3nfs01/data01/shome/threiten/QReg/dataMC-1/MTR/train_qRC_MC.py -q ${quantile} -W ${workDir} -F ${dataFrame} -D ${weightsDir} -i ${indWorker} -y ${year} -N ${N_evts}
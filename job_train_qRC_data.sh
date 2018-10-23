#!/bin/bash

echo SWITCH OFF DISPLAY
export DISPLAY=

source /mnt/t3nfs01/data01/shome/threiten/jupyter_env.sh

variable=$1
quantile=$2
workDir=$3
dataFrame=$4
weightsDir=$5
year=$6
N_evts=$7
EBEE=$8

python /mnt/t3nfs01/data01/shome/threiten/QReg/dataMC-1/MTR/train_qRC_data.py -v ${variable} -q ${quantile} -W ${workDir} -F ${dataFrame} -D ${weightsDir}  -y ${year} -N ${N_evts} -E ${EBEE}
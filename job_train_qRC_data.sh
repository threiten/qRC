#!/bin/bash

echo SWITCH OFF DISPLAY
export DISPLAY=

source /mnt/t3nfs01/data01/shome/threiten/jupyter_env.sh

config=$1
variable=$2
quantile=$3
N_evts=$4
EBEE=$5

python /mnt/t3nfs01/data01/shome/threiten/QReg/qRC/train_qRC_data.py -c ${config} -v ${variable} -q ${quantile} -N ${N_evts} -E ${EBEE}
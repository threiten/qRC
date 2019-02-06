#!/bin/bash

echo SWITCH OFF DISPLAY
export DISPLAY=

source /mnt/t3nfs01/data01/shome/threiten/jupyter_env.sh
export OMP_NUM_THREADS=4

config=$1
variable=$2
quantile=$3
N_evts=$4
EBEE=$5

if [ ! -z "$6" ]
then
    echo Doing split training!
    spl=$6
    python /mnt/t3nfs01/data01/shome/threiten/QReg/qRC/training/train_qRC_I_data.py -c ${config} -v ${variable} -q ${quantile} -N ${N_evts} -E ${EBEE} -s ${spl}
else
    python /mnt/t3nfs01/data01/shome/threiten/QReg/qRC/training/train_qRC_I_data.py -c ${config} -v ${variable} -q ${quantile} -N ${N_evts} -E ${EBEE}
fi
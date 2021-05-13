#!/usr/bin/env bash

config=$1
variable=$2
quantile=$3
N_evts=$4
EBEE=$5

#SBATCH -J ${detector}-${var}-${quantile}
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH -t 00:30:00

python $PWD/python_scripts/train_qRC_data.py -c ${config} -v ${variable} -q ${quantile} -N ${N_evts} -E ${EBEE}

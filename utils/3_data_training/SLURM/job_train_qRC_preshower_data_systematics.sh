#!/usr/bin/env bash

config=$1
variable=$2
quantile=$3
N_evts=$4
EBEE=$5
split=$6

#SBATCH -J ${detector}-${var}-${quantile}
#SBATCH -p quick
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH -t 03:00:00

python $PWD/python_scripts/train_qRC_I_preshower_data.py -c ${config} -v ${variable} -q ${quantile} -N ${N_evts} -E ${EBEE} -s ${split}

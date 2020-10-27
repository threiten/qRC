#!/bin/bash

#SBATCH --job-name=bayOpt_gpu

#SBATCH --account=gpu_gres

#SBATCH --nodes=1

#SBATCH --ntasks=5

#SBATCH --gres=gpu:1

var=$1
python /t3home/threiten/python/qRC/bayOpt/gpu/doBayOpt_qRC.py -v ${var} -n 5 2>&1 | tee /t3home/threiten/python/qRC/bayOpt/gpu/output_bayOpt_gpu_${var}_ReReco18.txt

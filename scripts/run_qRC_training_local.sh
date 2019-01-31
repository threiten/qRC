#!/bin/bash

i=0
j=0
workDir=$1
dataFrame_MC=$2
dataFrame_data=$3
weightsDir=$4
year=$5
N_evts=$6
N_jobs=$7


# for k in 1 2 3 4 5 6 7 8 9;
# do
#     for q in $( echo "scale=2; $k*0.1" | bc ) $( echo "scale=2; 0.05+$k*0.1" | bc ); #0.05 0.1;
#     do
# 	echo Starting training jobs for quantile $q on data
# 	python train_qRC_data.py -q $q -W ${workDir} -F ${dataFrame_data} -D ${weightsDir} -i $i -y ${year} -N ${N_evts} &
# 	# pids[${j}]=$!
# 	i=$(($i+6))
# 	pid=$!
# 	# j=$(($j+1))
# 	echo Submitting training jobs for quantile $q on MC
# 	python train_qRC_MC.py -q $q -W ${workDir} -F ${dataFrame_MC} -D ${weightsDir} -i $i -y ${year} -N ${N_evts} &
# 	# pids[${j}]=$!
# 	i=$(($i+6))
# 	# j=$(($j+1))
#     done
#     wait $!
#     wait pid
#     i=0
# done



for q in 0.05;# 0.25 0.3;
do
    echo Starting training jobs for quantile $q on data
    python train_qRC_data.py -q $q -W ${workDir} -F ${dataFrame_data} -D ${weightsDir} -i $i -y ${year} -N ${N_evts} &
    i=$(($i+6))
    echo Submitting training jobs for quantile $q on MC
    python train_qRC_MC.py -q $q -W ${workDir} -F ${dataFrame_MC} -D ${weightsDir} -i $i -y ${year} -N ${N_evts} &
    i=$(($i+6))
done

# for q in 0.05 0.45;
# do
#     echo Submitting training jobs for quantile $q on MC
#     python train_qRC_MC.py -q $q -W ${workDir} -F ${dataFrame_MC} -D ${weightsDir} -i $i -y ${year} -N ${N_evts} &
#     i=$(($i+6))
# done


# for pid in ${pids[*]};
# do
#     wait $pid
# done

# i=0
# j=0


# for pid in ${pids[*]};
# do
#     wait $pid
# done

# i=0
# j=0

# for q in 0.35 0.4 0.45 0.5;
# do
#     echo Starting training jobs for quantile $q on data
#     python train_qRC_data.py -q $q -W ${workDir} -F ${dataFrame_data} -D ${weightsDir} -i $i -y ${year} -N ${N_evts} &
#     pids[${j}]=$!
#     i=$(($i+6))
#     j=$(($j+1))
#     echo Submitting training jobs for quantile $q on MC
#     python train_qRC_MC.py -q $q -W ${workDir} -F ${dataFrame_MC} -D ${weightsDir} -i $i -y ${year} -N ${N_evts} &
#     pids[${j}]=$!
#     i=$(($i+6))
#     j=$(($j+1))
# done

# for pid in ${pids[*]};
# do
#     wait $pid
# done

# i=0
# j=0

# for q in 0.55 0.6 0.65 0.7;
# do
#     echo Starting training jobs for quantile $q on data
#     python train_qRC_data.py -q $q -W ${workDir} -F ${dataFrame_data} -D ${weightsDir} -i $i -y ${year} -N ${N_evts} &
#     pids[${j}]=$!
#     i=$(($i+6))
#     j=$(($j+1))
#     echo Submitting training jobs for quantile $q on MC
#     python train_qRC_MC.py -q $q -W ${workDir} -F ${dataFrame_MC} -D ${weightsDir} -i $i -y ${year} -N ${N_evts} &
#     pids[${j}]=$!
#     i=$(($i+6))
#     j=$(($j+1))
# done

# for pid in ${pids[*]};
# do
#     wait $pid
# done

# i=0
# j=0

# for q in 0.75 0.8 0.85 0.9;
# do
#     echo Starting training jobs for quantile $q on data
#     python train_qRC_data.py -q $q -W ${workDir} -F ${dataFrame_data} -D ${weightsDir} -i $i -y ${year} -N ${N_evts} &
#     pids[${j}]=$!
#     i=$(($i+6))
#     j=$(($j+1))
#     echo Submitting training jobs for quantile $q on MC
#     python train_qRC_MC.py -q $q -W ${workDir} -F ${dataFrame_MC} -D ${weightsDir} -i $i -y ${year} -N ${N_evts} &
#     pids[${j}]=$!
#     i=$(($i+6))
#     j=$(($j+1))
# done

# for pid in ${pids[*]};
# do
#     wait $pid
# done

# i=0
# j=0

# for q in 0.95;
# do
#     echo Starting training jobs for quantile $q on data
#     python train_qRC_data.py -q $q -W ${workDir} -F ${dataFrame_data} -D ${weightsDir} -i $i -y ${year} -N ${N_evts} &
#     pids[${j}]=$!
#     i=$(($i+6))
#     j=$(($j+1))
#     echo Submitting training jobs for quantile $q on MC
#     python train_qRC_MC.py -q $q -W ${workDir} -F ${dataFrame_MC} -D ${weightsDir} -i $i -y ${year} -N ${N_evts} &
#     pids[${j}]=$!
#     i=$(($i+6))
#     j=$(($j+1))
# done

# for pid in ${pids[*]};
# do
#     wait $pid
# done

# i=0
# j=0

# # done
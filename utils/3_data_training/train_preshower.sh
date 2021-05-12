for d in "EE";
do
    bash SLURM/run_data_training_preshower.sh config/config_qRC_training_preshower.yaml 4700000 ${d}
done

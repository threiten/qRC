for d in "EE";
do
    bash SLURM/run_data_training_preshower_systematics.sh config/config_qRC_training_preshower.yaml -1 ${d}
done

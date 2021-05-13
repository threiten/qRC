for d in "EB" "EE";
do
    bash SLURM/run_data_training_systematics.sh config/config_qRC_training_5M.yaml config/config_qRC_training_PhI_5M.yaml config/config_qRC_training_ChI_5M.yaml -1 ${d}
done

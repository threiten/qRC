# Train Data Shapes

Train 382 regressors (21 quantiles \* 9 variables \* 2 detector parts + 4 tail regressors).
Working in a SLURM cluster, running the following command:
```bash
$ ./train_all.sh
```
submits one job per regressor to be trained, using 4700000 events.

Remember to change both ```workDir``` and ```weightsDir``` inside the config files stored in ```config```, respectively to the path to the directory where the pandas dataframes are and to the output directory for the trained regressors.

To check which regressors are left, run:
```bash
$ python check_output_names.py -d path-to-weightsDir
```

# Convert Regressors Format

This step is necessary at the moment of writing (February 2021) because Flashgg takes as input regressors in ```xml``` format. 
Hopefully, with the new version of flashgg it will be possible to use the default output of XGBoost. For this reason, very little care is put into writing the code properly and the paths are hardcoded (if needed, change them directly).

- ```xgboost2tmva.py``` converts the regressors to the ```xml``` format
- ```convert_names.py``` changes the names from the training-specific convention to the flashgg one
- ```qRC_convert_to_xgb.ipynb``` produces a json file which has to be put in flashgg

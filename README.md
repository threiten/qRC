# Chained quantile regression
This repo contains the code to do data/MC correction using chained quantile regression and stochastic matching. 
The class `quantileRegression_chain` can be used to correct a set of continious variables differentially and 
while keeping their correlations. The class `quantileRegression_chain_disc` can be used to correct discontinious variables.
## Training BDTs for quantiles
To train the BDTs that will be used to extract the conditional pdf the functions `trainOnData` for data and `trainOnMC` 
for MC have to be used. For example:
```python
import quantileRegression_chain as qRegC
qRC = qRegC.quantileRegression_chain(year,EBEE,workDir,variables)
qRC.loadDataDF(df_name,0,stop,rsh,columns)
qRC.trainOnData(variable,weightsDir)
```

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import pickle as pkl
import gzip
import os
import ROOT as rt
import xgboost as xgb 

from joblib import delayed, Parallel, parallel_backend, register_parallel_backend

from IdMVAComputer import IdMvaComputer, helpComputeIdMva 
from Corrector import Corrector, applyCorrection
from quantileRegression_chain import quantileRegression_chain

class quantileRegression_chain_disc(quantileRegression_chain):

    def trainp0tclf(self,var,key,weightsDir ='weights_p0t',n_jobs=1):
        
        if key == 'MC':
            df = self.MC
        elif key == 'data':
            df = self.data
        else:
            raise KeyError('Please use data or MC')

        df['p0t_{}'.format(var)] = np.apply_along_axis(lambda x: 0 if x==0 else 1,0,df[var].reshape(1,-1))
        X = df.loc[:,self.kinrho + self.vars[:self.vars.index(var)]]
        Y = df['p0t_{}'.format(var)]
        clf = xgb.XGBClassifier(n_estimators=300,learning_rate=0.05,maxDepth=10,subsample=0.5,gamma=0, n_jobs=n_jobs)
        with parallel_backend(self.backend):
            clf.fit(X,Y)
        
        X_names = self.kinrho+self.vars[:self.vars.index(var)]
        Y_name = var
        dic = {'clf': clf, 'X': X_names, 'Y': Y_name}
        pkl.dump(dic,gzip.open('{}/{}/{}_clf_p0t_{}.pkl'.format(self.workDir,weightsDir,key,var),'wb'),protocol=pkl.HIGHEST_PROTOCOL)


    def trainTailRegressor(self,var,weightsDir='weights_tail',n_jobs=1):
        
        X = self.data.loc[:,self.kinrho+self.vars[:self.vars.index(var)]]
        Y = self.data[var]
        
        clf = xgb.XGBRegressor(n_estimators=300,gamma=0, maxDepth=10, n_jobs=n_jobs)
        with parallel_backend(self.backend):
            clf.fit(X,Y)
            
        X_names = self.kinrho+self.vars[:self.vars.index(var)]
        Y_name = var
        dic = {'clf': clf, 'X': X_names, 'Y': Y_name}
        pkl.dump(dic,gzip.open('{}/{}/data_reg_tail_{}.pkl'.format(self.workDir,weightsDir,var),'wb'),protocol=pkl.HIGHEST_PROTOCOL)

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
from quantileRegression_chain import quantileRegression_chain, trainClf
from Shifter import Shifter, applyShift

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
        pkl.dump(dic,gzip.open('{}/{}/{}_clf_p0t_{}_{}.pkl'.format(self.workDir,weightsDir,key,self.EBEE,var),'wb'),protocol=pkl.HIGHEST_PROTOCOL)


    def trainTailRegressor(self,var,weightsDir='weights_tail',n_jobs=1):
        
        X = self.data.loc[:,self.kinrho+self.vars[:self.vars.index(var)]]
        Y = self.data[var]
        
        clf = xgb.XGBRegressor(n_estimators=300,gamma=0, maxDepth=10, n_jobs=n_jobs)
        with parallel_backend(self.backend):
            clf.fit(X,Y)
            
        X_names = self.kinrho+self.vars[:self.vars.index(var)]
        Y_name = var
        dic = {'clf': clf, 'X': X_names, 'Y': Y_name}
        pkl.dump(dic,gzip.open('{}/{}/data_reg_tail_{}_{}.pkl'.format(self.workDir,weightsDir,self.EBEE,var),'wb'),protocol=pkl.HIGHEST_PROTOCOL)

    def loadTailRegressor(self,var,weightsDir):
        
        self.reg_tail = self.load_clf_safe(weightsDir,'data_reg_tail_{}_{}.pkl'.format(self.EBEE,var))

    def loadp0tclf(self,var,weightsDir):
        
        self.p0tclf_mc = self.load_clf_safe(weightsDir,'mc_clf_p0t_{}_{}.pkl'.format(self.EBEE,var))
        self.p0tclf_d = self.load_clf_safe(weightsDir,'data_clf_p0t_{}_{}.pkl'.format(self.EBEE,var))

    def shiftY(self,var,n_jobs=1):
        
        features = self.kinrho + ['{}_corr'.format(x) for x in self.vars[:self.vars.index(var)]]
        X = self.MC.loc[:,features]
        Y = self.MC[var]

        if X.isnull().values.any():
            raise KeyError('Correct {} first!'.format(self.vars[:self.vars.index(var)]))

        Y = Y.values.reshape(1,-1)
        Z = np.hstack([X,Y])
        
        print 'Shifting {} with input features {}'.format(var,features)

        with parallel_backend(self.backend):
            Y_shift = np.concatenate(Parallel(n_jobs=n_jobs, verbose=20)(delayed(applyShift)(self.p0tclf_mc,self.p0tclf_d,self.reg_tail,sli[:,:-1],sli[:,-1]) for sli in np.array_split(Z,n_jobs)))

        self.MC['{}_shift'.format(var)] = Y_shift

    def correctY(self,var,n_jobs=1,diz=None):
        
        self.shiftY(var,n_jobs=n_jobs)
        super(quantileRegression_chain_disc, self).correctY('{}_shift'.format(var), n_jobs=n_jobs, diz=True)

    def trainOnData(self,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):
        
        print 'Training quantile regressors on data'
        self._trainQuantiles('data_diz',var=var,maxDepth=maxDepth,minLeaf=minLeaf,weightsDir=weightsDir)
        
    def trainOnMC(self,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):
        
        print 'Training quantile regressors on MC'
        self._trainQuantiles('MC_diz',var=var,maxDepth=maxDepth,minLeaf=minLeaf,weightsDir=weightsDir)
    
    def trainAllMC(self,weightsDir,n_jobs=1):

        for var in self.vars:
            self.trainOnMC(var,weightsDir=weightsDir)
            self.loadTailRegressor(var,weightsDir=weightsDir)
            self.loadp0tclfs(var,weightsDir=weightsDir)
            self.loadClfs(var,weightsDir=weightsDir)
            self.correctY(var,n_jobs=n_jobs)

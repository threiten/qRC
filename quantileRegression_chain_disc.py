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
from Shifter2D import Shifter2D, apply2DShift

class quantileRegression_chain_disc(quantileRegression_chain):

    def trainp0tclf(self,var,key,weightsDir ='weights_qRC',n_jobs=1):
        
        if key == 'mc':
            df = self.MC
        elif key == 'data':
            df = self.data
        else:
            raise KeyError('Please use data or mc')

        features = self.kinrho

        df['p0t_{}'.format(var)] = np.apply_along_axis(lambda x: 0 if x==0 else 1,0,df[var].reshape(1,-1))
        X = df.loc[:,features].values
        Y = df['p0t_{}'.format(var)].values
        clf = xgb.XGBClassifier(n_estimators=300,learning_rate=0.05,maxDepth=10,subsample=0.5,gamma=0, n_jobs=n_jobs)
        with parallel_backend(self.backend):
            clf.fit(X,Y)
        
        X_names = features
        Y_name = var
        dic = {'clf': clf, 'X': X_names, 'Y': Y_name}
        pkl.dump(dic,gzip.open('{}/{}/{}_clf_p0t_{}_{}.pkl'.format(self.workDir,weightsDir,key,self.EBEE,var),'wb'),protocol=pkl.HIGHEST_PROTOCOL)
        
    def train3Catcfl(self,var1,var2,key,weightsDir='weights_qRC',n_jobs=1):

        if key == 'mc':
            df = self.MC
        elif key == 'data':
            df = self.data
        else:
            raise KeyError('Please use data or mc')
        
        features = self.kinrho

        df['ChIsoCat'] = self.get_class_3Cat(df[var1].values,df[var2].values)
        X = df.loc[:,features].values
        Y = df['ChIsoCat'].values
        clf = xgb.XGBClassifier(n_estimators=500,learning_rate=0.05,maxDepth=10,gamma=0,n_jobs=n_jobs)
        with parallel_backend(self.backend):
            clf.fit(X,Y)

        X_names = features
        Y_names = [var1,var2]
        dic = {'clf': clf, 'X': X_names, 'Y': Y_names}
        pkl.dump(dic,gzip.open('{}/{}/{}_clf_3Cat_{}_{}_{}.pkl'.format(self.workDir,weightsDir,key,self.EBEE,var1,var2),'wb'),protocol=pkl.HIGHEST_PROTOCOL)
        
    def get_class_3Cat(self,x,y):
        return [0 if x[i]==0 and y[i]==0 else (1 if x[i]==0 and y[i]>0 else 2) for i in range(len(x))]

    def load3Catclf(self,varrs,weightsDir='weights_qRC'):
        
        self.TCatclf_mc = self.load_clf_safe(varrs,weightsDir,'mc_clf_3Cat_{}_{}_{}.pkl'.format(self.EBEE,varrs[0],varrs[1]),self.kinrho)
        self.TCatclf_d = self.load_clf_safe(varrs,weightsDir,'data_clf_3Cat_{}_{}_{}.pkl'.format(self.EBEE,varrs[0],varrs[1]),self.kinrho)

    def trainTailRegressors(self,var,weightsDir='weights_qRC'):
        
        features = self.kinrho+['{}'.format(x) for x in self.vars if not x == var]
        X = self.MC.query('{}!=0'.format(var)).loc[:,features].values
        Y = self.MC.query('{}!=0'.format(var))[var].values
        
        with parallel_backend(self.backend):
            Parallel(n_jobs=len(self.quantiles),verbose=20)(delayed(trainClf)(q,5,500,X,Y,save=True,outDir='{}/{}'.format(self.workDir,weightsDir),name='mc_weights_tail_{}_{}_{}'.format(self.EBEE,var,str(q).replace('.','p')),X_names=features,Y_name=var) for q in self.quantiles)
            
    def loadTailRegressors(self,varrs,weightsDir):
        
        self.tail_clfs_mc = {}
        self.tail_clfs_mc[varrs[0]] = [self.load_clf_safe(varrs[0], weightsDir,'mc_weights_tail_{}_{}_{}.pkl'.format(self.EBEE,varrs[0],str(q).replace('.','p')),self.kinrho+[varrs[1]]) for q in self.quantiles]
        self.tail_clfs_mc[varrs[1]] = [self.load_clf_safe(varrs[1], weightsDir,'mc_weights_tail_{}_{}_{}.pkl'.format(self.EBEE,varrs[1],str(q).replace('.','p')),self.kinrho+[varrs[0]]) for q in self.quantiles]

    def loadp0tclf(self,var,weightsDir):
        
        self.p0tclf_mc = self.load_clf_safe(var, weightsDir,'mc_clf_p0t_{}_{}.pkl'.format(self.EBEE,var))
        self.p0tclf_d = self.load_clf_safe(var, weightsDir,'data_clf_p0t_{}_{}.pkl'.format(self.EBEE,var))

    def shiftY(self,var,n_jobs=1):
        
        features = self.kinrho
        X = self.MC.loc[:,features]
        Y = self.MC[var]

        Y = Y.values.reshape(-1,1)
        Z = np.hstack([X,Y])
        
        print 'Shifting {} with input features {}'.format(var,features)

        with parallel_backend(self.backend):
            Y_shift = np.concatenate(Parallel(n_jobs=n_jobs, verbose=20)(delayed(applyShift)(self.p0tclf_mc,self.p0tclf_d,self.clfs_mc,sli[:,:-1],sli[:,-1]) for sli in np.array_split(Z,n_jobs)))

        self.MC['{}_shift'.format(var)] = Y_shift

    def shiftY2D(self,var1,var2,n_jobs=1):
        
        features = self.kinrho
        X = self.MC.loc[:,features]
        Y = self.MC.loc[:,[var1,var2]]

        if X.isnull().values.any():
            raise KeyError('Correct {} first!'.format(self.vars[:self.vars.index(var)]))

        Y = Y.values.reshape(-1,2)
        Z = np.hstack([X,Y])
        
        with parallel_backend(self.backend):
            Y_shift = np.concatenate(Parallel(n_jobs=n_jobs, verbose=20)(delayed(apply2DShift)(self.TCatclf_mc,self.TCatclf_d,self.tail_clfs_mc[varrs[0]],self.tail_clfs_mc[varrs[1]],sli[:,:-2],sli[:,-2:]) for sli in np.array_split(Z,n_jobs)))
            
        self.MC['{}_shift'.format(var1)] = Y_shift[:,0]
        self.MC['{}_shift'.format(var2)] = Y_shift[:,1]
        
    def correctY(self,var,n_jobs=1,diz=None):
        
        if len(self.vars)==1:
            self.shiftY(var,n_jobs=n_jobs)
        elif len(self.vars)>1 and '{}_shift'.format(var) not in self.MC.columns:
            self.shiftY2D(var1=self.vars[0],var2=self.vars[1],n_jobs=n_jobs)
        super(quantileRegression_chain_disc, self).correctY('{}_shift'.format(var), n_jobs=n_jobs, diz=True)

    def trainOnData(self,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):
        
        print 'Training quantile regressors on data'
        self._trainQuantiles('data_diz',var=var,maxDepth=maxDepth,minLeaf=minLeaf,weightsDir=weightsDir)
        
    def trainOnMC(self,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):
        
        print 'Training quantile regressors on MC'
        self._trainQuantiles('mc_diz',var=var,maxDepth=maxDepth,minLeaf=minLeaf,weightsDir=weightsDir)

    def trainFinalRegression(self,var,weightsDir,n_jobs=1):
        super(quantileRegression_chain_disc,self).trainFinalRegression(var,weightsDir,diz=True,n_jobs=n_jobs)
    
    def trainFinalTailRegressor(self,var,weightsDir,weightsDirIn,n_jobs=1):
        

        df = self.MC.query('{}!=0'.format(var))

        if len(self.vars) == 1:
            self.loadClfs(var,weightsDirIn)
            df['cdf_{}'.format(var)] = self._getCondCDF(df,self.clfs_mc,self.kinrho,var)
        elif len(self.vars) > 1:
            self.loadTailRegressors(self.vars,weightsDirIn)
            df['cdf_{}'.format(var)] = self._getCondCDF(df,self.tail_clfs_mc[var],self.kinrho+[x for x in self.vars if not x == var],var)
            
        features = self.kinrho + [x for x in self.vars if not x == var] + ['cdf_{}'.format(var)]
        X = df.loc[:,features]
        Y = df[var]
        
        print 'Training final tail regressor with features {} for {}'.format(features,var)
        clf = xgb.XGBRegressor(n_estimators=1000, maxDepth=10, gamma=0, n_jobs=n_jobs)
        clf.fit(X,Y)

        name = 'weights_finalTailRegressor_{}_{}'.format(self.EBEE,var)
        dic = {'clf': clf, 'X': features, 'Y': var}
        pkl.dump(dic,gzip.open('{}/{}/{}.pkl'.format(self.workDir,weightsDir,name),'wb'),protocol=pkl.HIGHEST_PROTOCOL)

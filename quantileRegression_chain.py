import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import pickle as pkl
import gzip
import os
import ROOT as rt

from joblib import delayed, Parallel, parallel_backend, register_parallel_backend

from IdMVAComputer import IdMvaComputer, helpComputeIdMva 
from Corrector import Corrector, applyCorrection
#from sklearn.externals.joblib import Parallel, parallel_backend, register_parallel_backend


class quantileRegression_chain(object):

    def __init__(self,year,EBEE,workDir,varrs):

        self.year = year
        self.workDir = workDir
        self.kinrho = ['probePt','probeScEta','probePhi','rho']
        self.vars = varrs
        self.quantiles = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
        self.backend = 'loky'
        self.EBEE = EBEE


    def _loadDF(self, h5name, start=0, stop=-1, rndm=12345, rsh=False, columns=None):
        
        if rsh:
            df = pd.read_hdf('{}/{}'.format(self.workDir,h5name), 'df', columns=columns)
        else:
            df = pd.read_hdf('{}/{}'.format(self.workDir,h5name), 'df', columns=columns, start=start, stop=stop)
        
        index = np.array(df.index)
        if rsh:
            print 'Reshuffling events'
            np.random.seed(rndm)
            np.random.shuffle(index)
            df = df.ix[index]
            df.reset_index(drop=True, inplace=True)

        if stop == -1:
            stop = df.index.size + 1

        df = df[start:stop]

        if self.EBEE == 'EB':
            df = df.query('probeScEta>-1.4442 and probeScEta<1.4442')
        elif self.EBEE == 'EE':
            df = df.query('probeScEta<-1.556 or probeScEta>1.556')

        if df.index.size==0:
            raise ValueError('Wrong dataframe selected!')

        return df

    def loadMCDF(self,h5name,start=0,stop=-1,rndm=12345,rsh=False,columns=None):
        
        print 'Loading MC Dataframe from: {}/{}'.format(self.workDir,h5name)
        self.MC = self._loadDF(h5name,start,stop,rndm,rsh,columns)
        
    def loadDataDF(self,h5name,start=0,stop=-1,rndm=12345,rsh=False,columns=None):
        
        print 'Loading data Dataframe from: {}/{}'.format(self.workDir,h5name)
        self.data = self._loadDF(h5name,start,stop,rndm,rsh,columns)

    def trainOnData(self,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):
        
        print 'Training quantile regressors on data'
        self._trainQuantiles('data',var=var,maxDepth=maxDepth,minLeaf=minLeaf,weightsDir=weightsDir)
        
    def trainOnMC(self,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):
        
        print 'Training quantile regressors on MC'
        self._trainQuantiles('MC',var=var,maxDepth=maxDepth,minLeaf=minLeaf,weightsDir=weightsDir)

    def _trainQuantiles(self,key,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):

        if var not in self.vars:
            raise ValueError('{} has to be one of {}'.format(var, vars))
        
        if key.startswith('MC'):
            features = self.kinrho + ['{}_corr'.format(x) for x in self.vars[:self.vars.index(var)]]
        elif key.startswith('data'):
            features = self.kinrho + self.vars[:self.vars.index(var)]
        else:
            raise KeyError('Key needs to specify if data or MC')

        if 'diz' in key:
            X = self.data.query('{}!=0'.format(var)).loc[:,features]
            Y = self.data.query('{}!=0'.format(var))[var]
        else:
            X = self.MC.loc[:,features]
            Y = self.MC[var]

        name_key = 'data' if 'data' in key else 'mc'

        with parallel_backend(self.backend):
            Parallel(n_jobs=len(self.quantiles),verbose=20)(delayed(trainClf)(q,maxDepth,minLeaf,X,Y,save=True,outDir='{}/{}'.format(self.workDir,weightsDir),name='{}_weights_{}_{}_{}'.format(name_key,self.EBEE,var,str(q).replace('.','p')),X_names=features,Y_name=var) for q in self.quantiles)

        
    def correctY(self, var, n_jobs=1, diz=False):
        
        features = self.kinrho + ['{}_corr'.format(x) for x in self.vars[:self.vars.index(var)]]
        X = self.MC.loc[:,features]
        Y = self.MC[var]
        
        if X.isnull().values.any():
            raise KeyError('Correct {} first!'.format(self.vars[:self.vars.index(var)]))

        print "Features: X = ", features, " target y = ", var
        
        Y = Y.values.reshape(-1,1)
        Z = np.hstack([X,Y])

        with parallel_backend(self.backend):
            Ycorr = np.concatenate(Parallel(n_jobs=n_jobs,verbose=20)(delayed(applyCorrection)(self.clfs_mc,self.clfs_d,ch[:,:-1],ch[:,-1],diz=diz) for ch in np.array_split(Z,n_jobs) ) )

        self.MC['{}_corr'.format(var)] = Ycorr

    def trainAllMC(self,weightsDir,n_jobs=1):
        
        for var in self.vars:
            self.trainOnMC(var,weightsDir=weightsDir)
            self.loadClfs(var,weightsDir)
            self.correctY(var,n_jobs=n_jobs)
            
    def loadClfs(self, var, weightsDir):
        
        self.clfs_mc = [self.load_clf_safe(weightsDir, 'mc_weights_{}_{}_{}.pkl'.format(self.EBEE,var,str(q).replace('.','p'))) for q in self.quantiles]
        self.clfs_d = [self.load_clf_safe(weightsDir,'data_weights_{}_{}_{}.pkl'.format(self.EBEE,var,str(q).replace('.','p'))) for q in self.quantiles]
        
    def load_clf_safe(self,weightsDir,name):
        
        clf = pkl.load(gzip.open('{}/{}/{}'.format(self.workDir,weightsDir,name)))
        if name.startswith('mc'):
            X_name = self.kinrho + ['{}_corr'.format(x) for x in self.vars[:self.vars.index(var)]]
        elif name.startswith('data'):
            X_name = self.kinrho +  self.vars[:self.vars.index(var)]
        else:
            raise NameError('name has to start with data or mc')
   
        if clf['X'] != X_name or clf['Y'] != var:
            raise ValueError('{}/{}/{} was not trained with the right order of Variables!'.format(self.workDir,weightsDir,name))
        else:
            return clf['clf']
    
    def computeIdMvas(self,mvas,weights,key,n_jobs=1,leg2016=False):
      weightsEB,weightsEE = weights
      for name,tpC,correctedVariables in mvas:
         self.computeIdMva(name,weightsEB,weightsEE,key,correctedVariables,tpC,leg2016,n_jobs)

    def computeIdMva(self,name,weightsEB,weightsEE,key,correctedVariables,tpC,leg2016,n_jobs):
        stride = self.MC.index.size / n_jobs
        print("Computing %s, correcting %s, stride %s" % (name,correctedVariables,stride) )
        if key == 'MC':
            with parallel_backend(self.backend):
                Y = np.concatenate(Parallel(n_jobs=n_jobs,verbose=20)(delayed(helpComputeIdMva)(weightsEB,weightsEE,correctedVariables,self.MC[ch:ch+stride],tpC, leg2016) for ch in xrange(0,self.MC.index.size,stride)))
            self.MC[name] = Y
        elif key == 'data':
            with parallel_backend(self.backend):
                Y = np.concatenate(Parallel(n_jobs=n_jobs,verbose=20)(delayed(helpComputeIdMva)(weightsEB,weightsEE,correctedVariables,self.data[ch:ch+stride],tpC, leg2016) for ch in xrange(0,self.data.index.size,stride)))
            self.data[name] = Y


    def setupJoblib(self,ipp_profile='default',sel_workers=None):
        
        import ipyparallel as ipp
        global joblib_rc,joblib_view
        joblib_rc = ipp.Client(profile=ipp_profile)
        joblib_view = joblib_rc.load_balanced_view(sel_workers)
        joblib_view.register_joblib_backend()
        self.backend = 'ipyparallel'


def trainClf(alpha,maxDepth,minLeaf,X,Y,save=False,outDir=None,name=None,X_names=None,Y_name=None):
    
    clf = GradientBoostingRegressor(loss='quantile', alpha=alpha,
                                    n_estimators=500, max_depth=maxDepth,
                                    learning_rate=.1, min_samples_leaf=minLeaf,
                                    min_samples_split=minLeaf)
        
    clf.fit(X,Y)

    if save and (outDir is None or name is None or X_names is None or Y_name is None):
        raise TypeError('outDir, name, X_names and Y_name must not be NoneType if save=True')
    if save:
        print 'Saving clf trained with features {} for {} to {}/{}.pkl'.format(X_names,Y_name,outDir,name)
        dic = {'clf': clf, 'X': X_names, 'Y': Y_name}
        pkl.dump(dic,gzip.open('{}/{}.pkl'.format(outDir,name),'wb'),protocol=pkl.HIGHEST_PROTOCOL)
    
    return clf

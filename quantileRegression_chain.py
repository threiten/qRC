import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import pickle as pkl
import gzip
import os
import ROOT as rt
from root_pandas import read_root

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
        self.branches = ['probeScEta','probeEtaWidth','probeR9','weight','probeSigmaRR','tagScPreshowerEnergy','probePass_invEleVeto','tagChIso03','probeChIso03','probeS4','tagR9','tagPhiWidth','probePt','tagSigmaRR','probePhiWidth','probeChIso03worst','puweight','tagEleMatch','tagPhi','probeScEnergy','nvtx','probePhoIso','tagPhoIso','run','tagScEta','probeEleMatch','probeCovarianceIeIp','tagPt','rho','tagS4','tagSigmaIeIe','tagCovarianceIpIp','tagCovarianceIeIp','tagScEnergy','tagChIso03worst','probeSigmaIeIe','probePhi','mass','probeCovarianceIpIp','tagEtaWidth','probeScPreshowerEnergy']

        if year == '2016':
            self.branches = self.branches + ['probePass_invEleVeto','probeCovarianceIetaIphi','probeCovarianceIphiIphi','probeCovarianceIetaIphi','probeCovarianceIphiIphi']
            self.branches.remove('probeCovarianceIeIp')
            self.branches.remove('probeCovarianceIpIp')
            self.branches.remove('tagCovarianceIeIp')
            self.branches.remove('tagCovarianceIpIp')

        self.ptmin  =  25.
        self.ptmax  =  150.
        self.etamin = -2.5
        self.etamax =  2.5
        self.phimin = -3.14
        self.phimax =  3.14

    def loadROOT(self,path,tree,outname,cut=None,split=None,rsh=True,rndm=12345):
        
        if 'Data' in tree:
            df = read_root(path,tree,columns=self.branches)
        elif self.year == '2016':
            df = read_root(path,tree,columns=self.branches+['probePhoIso_corr'])

        index = np.array(df.index)
        
        print 'Reshuffling events'
        np.random.seed(rndm)
        np.random.shuffle(index)
        df = df.ix[index]

        df = df.query('probePt>@self.ptmin and probePt<@self.ptmax and probeScEta>@self.etamin and probeScEta<@self.etamax and probePhi>@self.phimin and probePhi<@self.phimax')

        if self.EBEE == 'EB':
            print 'Selecting events from EB'
            df = df.query('probeScEta>-1.4442 and probeScEta<1.4442')
        elif self.EBEE == 'EE':
            print 'Selecting events from EE'
            df = df.query('probeScEta<-1.556 or probeScEta>1.556')
        
        
        if cut is not None:
            print 'Applying cut {}'.format(cut)
            df = df.query(cut)
        
        df.reset_index(drop=True, inplace=True)

        if self.year=='2016' and not 'Data' in tree:
            df['probePhoIso_corr_sto'] = df['probePhoIso_corr']
            
        if split is not None:
            print 'Splitting dataframe in train and test sample. Split size is at {}%'.format(int(split*100))
            df_train = df[0:int(split*df.index.size)]
            df_test = df[int(split*df.index.size):]
            print 'Number of events in training dataframe {}. Saving to {}/{}_(train/test).h5'.format(df_train.index.size,self.workDir,outname)
            df_train.to_hdf('{}/{}_train.h5'.format(self.workDir,outname),'df',mode='w',format='t')
            df_test.to_hdf('{}/{}_test.h5'.format(self.workDir,outname),'df',mode='w',format='t')
        else:
            print 'Number of events in dataframe {}. Saving to {}/{}.h5'.format(df.index.size,self.workDir,outname)
            df.to_hdf('{}/{}.h5'.format(self.workDir,outname),'df',mode='w',format='t')
            
        return df
        
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
        self._trainQuantiles('mc',var=var,maxDepth=maxDepth,minLeaf=minLeaf,weightsDir=weightsDir)

    def _trainQuantiles(self,key,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):

        if var not in self.vars+[x+'_shift' for x in self.vars]:
            raise ValueError('{} has to be one of {}'.format(var, self.vars))
        
        if 'diz' in key:
            querystr = '{}!=0'.format(var)
        else:
            querystr = '{}=={}'.format(var,var)

        if key.startswith('mc'):
            features = self.kinrho + ['{}_corr'.format(x) for x in self.vars[:self.vars.index(var)]]
            X = self.MC.query(querystr).loc[:,features]
            Y = self.MC.query(querystr)[var]
        elif key.startswith('data'):
            features = self.kinrho + self.vars[:self.vars.index(var)]
            X = self.data.query(querystr).loc[:,features]
            Y = self.data.query(querystr)[var]
        else:
            raise KeyError('Key needs to specify if data or mc')

        name_key = 'data' if 'data' in key else 'mc'

        with parallel_backend(self.backend):
            Parallel(n_jobs=len(self.quantiles),verbose=20)(delayed(trainClf)(q,maxDepth,minLeaf,X,Y,save=True,outDir='{}/{}'.format(self.workDir,weightsDir),name='{}_weights_{}_{}_{}'.format(name_key,self.EBEE,var,str(q).replace('.','p')),X_names=features,Y_name=var) for q in self.quantiles)

        
    def correctY(self, var, n_jobs=1, diz=False):
        
        features = self.kinrho + ['{}_corr'.format(x) for x in self.vars[:self.vars.index(var[:var.find('_')])]]
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
        
        self.clfs_mc = [self.load_clf_safe(var, weightsDir, 'mc_weights_{}_{}_{}.pkl'.format(self.EBEE,var,str(q).replace('.','p'))) for q in self.quantiles]
        self.clfs_d = [self.load_clf_safe(var, weightsDir,'data_weights_{}_{}_{}.pkl'.format(self.EBEE,var,str(q).replace('.','p'))) for q in self.quantiles]
        
    def load_clf_safe(self,var,weightsDir,name):
        
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
        if key == 'mc':
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

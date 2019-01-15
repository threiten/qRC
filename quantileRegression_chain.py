import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
import pandas as pd
import pickle as pkl
import xgboost as xgb
import gzip
import os
import ROOT as rt
from root_pandas import read_root

from joblib import delayed, Parallel, parallel_backend, register_parallel_backend

from IdMVAComputer import IdMvaComputer, helpComputeIdMva
from eleIdMVAComputer import eleIdMvaComputer, helpComputeEleIdMva
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
        self.branches = ['probeScEta','probeEtaWidth','probeR9','weight','probeSigmaRR','tagScPreshowerEnergy','probePass_invEleVeto','tagChIso03','probeChIso03','probeS4','tagR9','tagPhiWidth','probePt','tagSigmaRR','probePhiWidth','probeChIso03worst','puweight','tagEleMatch','tagPhi','probeScEnergy','nvtx','probePhoIso','tagPhoIso','run','tagScEta','probeEleMatch','probeCovarianceIeIp','tagPt','rho','tagS4','tagSigmaIeIe','tagCovarianceIpIp','tagCovarianceIeIp','tagScEnergy','tagChIso03worst','probeSigmaIeIe','probePhi','mass','probeCovarianceIpIp','tagEtaWidth','probeScPreshowerEnergy','probeHoE','probeFull5x5_e1x5','probeFull5x5_e5x5','probeNeutIso']

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

    def loadROOT(self,path,tree,outname,cut=None,split=None,rndm=12345):
        
        if self.year == '2016' and 'Data' not in tree:
            df = read_root(path,tree,columns=self.branches+['probePhoIso_corr'])
        else:
            df = read_root(path,tree,columns=self.branches)
        
        print 'Dataframe with columns {}'.format(df.columns)
        index = np.array(df.index)
        
        print 'Reshuffling events'
        np.random.seed(rndm)
        np.random.shuffle(index)
        df = df.ix[index]

        df.query('probePt>@self.ptmin and probePt<@self.ptmax and probeScEta>@self.etamin and probeScEta<@self.etamax and probePhi>@self.phimin and probePhi<@self.phimax',inplace=True)

        if self.EBEE == 'EB':
            print 'Selecting events from EB'
            df.query('probeScEta>-1.4442 and probeScEta<1.4442',inplace=True)
        elif self.EBEE == 'EE':
            print 'Selecting events from EE'
            df.query('probeScEta<-1.556 or probeScEta>1.556',inplace=True)
        
        
        if cut is not None:
            print 'Applying cut {}'.format(cut)
            df.query(cut,inplace=True)
        
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

        if self.EBEE == 'EB' and df[abs(df['probeScEta']>1.4442)].index.size>0:
            df.query('probeScEta>-1.4442 and probeScEta<1.4442',inplace=True)
        elif self.EBEE == 'EE' and df[abs(df['probeScEta']<1.556)].index.size>0:
            df.query('probeScEta<-1.556 or probeScEta>1.556',inplace=True)
        
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
        
        self._trainQuantiles('data',var=var,maxDepth=maxDepth,minLeaf=minLeaf,weightsDir=weightsDir)
        
    def trainOnMC(self,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):
        
        self._trainQuantiles('mc',var=var,maxDepth=maxDepth,minLeaf=minLeaf,weightsDir=weightsDir)

    def _trainQuantiles(self,key,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):

        if var not in self.vars+['{}_shift'.format(x) for x in self.vars]:
            raise ValueError('{} has to be one of {}'.format(var, self.vars))
        
        # if 'diz' in key:
        #     querystr = '{}!=0'.format(var)
        # else:
        #     querystr = '{0}=={0}'.format(var)

        if key.startswith('mc'):
            features = self.kinrho + ['{}_corr'.format(x) for x in self.vars[:self.vars.index(var)]]
            if 'diz' in key:
                X = self.MC.loc[self.MC[var]!=0,features]
                Y = self.MC.loc[self.MC[var]!=0,var]
            else:
                X = self.MC.loc[:,features]
                Y = self.MC.loc[:,var]

        elif key.startswith('data'):
            features = self.kinrho + self.vars[:self.vars.index(var)]
            if 'diz' in key:
                X = self.data.loc[self.data[var]!=0,features]
                Y = self.data.loc[self.data[var]!=0,var]
            else:
                X = self.data.loc[:,features]
                Y = self.data.loc[:,var]
        else:
            raise KeyError('Key needs to specify if data or mc')

        name_key = 'data' if 'data' in key else 'mc'

        print 'Training quantile regrssion on {} for {} with features {}'.format(key,var,features)

        with parallel_backend(self.backend):
            Parallel(n_jobs=len(self.quantiles),verbose=20)(delayed(trainClf)(q,maxDepth,minLeaf,X,Y,save=True,outDir='{}/{}'.format(self.workDir,weightsDir),name='{}_weights_{}_{}_{}'.format(name_key,self.EBEE,var,str(q).replace('.','p')),X_names=features,Y_name=var) for q in self.quantiles)

        
    def correctY(self, var, n_jobs=1, diz=False):
        
        var_raw = var[:var.find('_')] if '_' in var else var
        features = self.kinrho + ['{}_corr'.format(x) for x in self.vars[:self.vars.index(var_raw)]]
        X = self.MC.loc[:,features]
        Y = self.MC[var]
        
        if X.isnull().values.any():
            raise KeyError('Correct {} first!'.format(self.vars[:self.vars.index(var)]))

        print "Features: X = ", features, " target y = ", var
        
        Y = Y.values.reshape(-1,1)
        Z = np.hstack([X,Y])

        with parallel_backend(self.backend):
            Ycorr = np.concatenate(Parallel(n_jobs=n_jobs,verbose=20)(delayed(applyCorrection)(self.clfs_mc,self.clfs_d,ch[:,:-1],ch[:,-1],diz=diz) for ch in np.array_split(Z,n_jobs) ) )

        self.MC['{}_corr'.format(var_raw)] = Ycorr

    def trainFinalRegression(self,var,weightsDir,diz=False,n_jobs=1):
        
        robSca = RobustScaler()
        features = self.kinrho + self.vars
        target = '{}_corr_diff_scale'.format(var)

        if diz:
            querystr = '{}!=0 and {}_corr!=0'.format(var,var)
        else:
            querystr = '{}=={}'.format(var,var)

        df = self.MC.query(querystr)

        df['{}_corr_diff_scale'.format(var)] = robSca.fit_transform(np.array(df['{}_corr'.format(var)] - df[var]).reshape(-1,1))
        pkl.dump(robSca,gzip.open('{}/{}/scaler_mc_{}_{}_corr_diff.pkl'.format(self.workDir,weightsDir,self.EBEE,var),'wb'),protocol=pkl.HIGHEST_PROTOCOL)

        X = df.loc[:,features].values
        Y = df[target].values

        clf = xgb.XGBRegressor(n_estimators=1000, maxDepth=10, gamma=0, n_jobs=n_jobs, base_score=0.)
        clf.fit(X,Y)

        name = 'weights_finalRegressor_{}_{}'.format(self.EBEE,var)
        print 'Saving final regrssion trained with features {} for {} to {}/{}.pkl'.format(features,'{}_corr_diff_scale'.format(var),weightsDir,name)
        dic = {'clf': clf, 'X': features, 'Y': '{}_corr_diff_scale'.format(var)}
        pkl.dump(dic,gzip.open('{}/{}/{}.pkl'.format(self.workDir,weightsDir,name),'wb'),protocol=pkl.HIGHEST_PROTOCOL)
        
    def loadFinalRegression(self,var,weightsDir):
        
        self.finalReg = self.load_clf_safe(var,weightsDir,'weights_finalRegressor_{}_{}.pkl'.format(self.EBEE,var),self.kinrho+self.vars,'{}_corr_diff_scale'.format(var))

    def loadScaler(self,var,weightsDir):
        
        self.scaler = pkl.load(gzip.open('{}/{}/scaler_mc_{}_{}_corr_diff.pkl'.format(self.workDir,weightsDir,self.EBEE,var)))

    def applyFinalRegression(self,var,diz=False):
        
        var_raw = var[:var.find('_')] if '_' in var else var
        features = self.kinrho + self.vars
        if diz:
            X = self.MC.loc[self.MC[var] != 0,features].values
            self.MC.loc[self.MC[var] != 0,'{}_corr_1Reg'.format(var_raw)] = self.MC.loc[self.MC[var] != 0,var] + self.scaler.inverse_transform(self.finalReg.predict(X))
            self.MC.loc[self.MC[var] == 0,'{}_corr_1Reg'.format(var_raw)] = 0
        else:
            X = self.MC.loc[:,features].values
            self.MC['{}_corr_1Reg'.format(var)] = self.MC[var] + self.scaler.inverse_transform(self.finalReg.predict(X))
        
    def trainAllMC(self,weightsDir,n_jobs=1):
        
        for var in self.vars:
            self.trainOnMC(var,weightsDir=weightsDir)
            self.loadClfs(var,weightsDir)
            self.correctY(var,n_jobs=n_jobs)
            
    def loadClfs(self, var, weightsDir):
        
        self.clfs_mc = [self.load_clf_safe(var, weightsDir, 'mc_weights_{}_{}_{}.pkl'.format(self.EBEE,var,str(q).replace('.','p'))) for q in self.quantiles]
        self.clfs_d = [self.load_clf_safe(var, weightsDir,'data_weights_{}_{}_{}.pkl'.format(self.EBEE,var,str(q).replace('.','p'))) for q in self.quantiles]
        
    def load_clf_safe(self,var,weightsDir,name,X_name=None,Y_name=None):
        
        clf = pkl.load(gzip.open('{}/{}/{}'.format(self.workDir,weightsDir,name)))

        if X_name is None:
            if name.startswith('mc'):
                X_name = self.kinrho + ['{}_corr'.format(x) for x in self.vars[:self.vars.index(var)]]
            elif name.startswith('data'):
                X_name = self.kinrho +  self.vars[:self.vars.index(var)]
            else:
                raise NameError('name has to start with data or mc')
           
        if Y_name is None:
            Y_name=var
            
        if clf['X'] != X_name or clf['Y'] != Y_name:
            raise ValueError('{}/{}/{} was not trained with the right order of Variables! Got {}, stored in file {}'.format(self.workDir,weightsDir,name,X_name,clf['X']))
        else:
            return clf['clf']
        
    def _getCondCDF(self,df,clfs,features,var):
        
        qtls_names = ['q{}_{}'.format(str(self.quantiles[i]).replace('0.','p'),var) for i in range(len(self.quantiles))]
        X = df.loc[:,features]
        if not all(qtls_names) in df.columns:
            mcqtls = [clf.predict(X) for clf in clfs]
            for i in range(len(self.quantiles)):
                df[qtls_names[i]] = mcqtls[i]

        return df.loc[:,[var] +qtls_names].apply(self._getCDFval,1,raw=True)
        
    def _getCDFval(self,row):
        
        Y = row[0]
        qtls = np.array(row[1:].values,dtype=float)
        bins = self.quantiles

        ind = np.searchsorted(qtls,Y)

        if Y<=qtls[0]:
            return np.random.uniform(0,0.01)
        elif Y>qtls[-1]:
            return np.random.uniform(0.99,1)

        return np.interp(Y,qtls[ind-1:ind+1],bins[ind-1:ind+1])
        
    def computeIdMvas(self,mvas,weights,key,n_jobs=1,leg2016=False):
      weightsEB,weightsEE = weights
      for name,tpC,correctedVariables in mvas:
         self.computeIdMva(name,weightsEB,weightsEE,key,correctedVariables,tpC,leg2016,n_jobs)

    def computeIdMva(self,name,weightsEB,weightsEE,key,correctedVariables,tpC,leg2016,n_jobs):
        if key=='mc':
            stride = self.MC.index.size / n_jobs
        elif key=='data':
            stride = self.data.index.size / n_jobs
        print("Computing %s, correcting %s, stride %s" % (name,correctedVariables,stride) )
        if key == 'mc':
            with parallel_backend(self.backend):
                Y = np.concatenate(Parallel(n_jobs=n_jobs,verbose=20)(delayed(helpComputeIdMva)(weightsEB,weightsEE,correctedVariables,self.MC[ch:ch+stride],tpC, leg2016) for ch in xrange(0,self.MC.index.size,stride)))
            self.MC[name] = Y
        elif key == 'data':
            with parallel_backend(self.backend):
                Y = np.concatenate(Parallel(n_jobs=n_jobs,verbose=20)(delayed(helpComputeIdMva)(weightsEB,weightsEE,correctedVariables,self.data[ch:ch+stride],tpC, leg2016) for ch in xrange(0,self.data.index.size,stride)))
            self.data[name] = Y

    def computeEleIdMvas(self,mvas,weights,key,n_jobs=1,leg2016=False):
      weightsEB1,weightsEB2,weightsEE = weights
      for name,tpC,correctedVariables in mvas:
         self.computeEleIdMva(name,weightsEB1,weightsEB2,weightsEE,key,correctedVariables,tpC,leg2016,n_jobs)

    def computeEleIdMva(self,name,weightsEB1,weightsEB2,weightsEE,key,correctedVariables,tpC,leg2016,n_jobs):
        if key=='mc':
            stride = self.MC.index.size / n_jobs
        elif key=='data':
            stride = self.data.index.size / n_jobs
        print("Computing %s, correcting %s, stride %s" % (name,correctedVariables,stride) )
        if key == 'mc':
            with parallel_backend(self.backend):
                Y = np.concatenate(Parallel(n_jobs=n_jobs,verbose=20)(delayed(helpComputeEleIdMva)(weightsEB1,weightsEB2,weightsEE,correctedVariables,self.MC[ch:ch+stride],tpC, leg2016) for ch in xrange(0,self.MC.index.size,stride)))
            self.MC[name] = Y
        elif key == 'data':
            with parallel_backend(self.backend):
                Y = np.concatenate(Parallel(n_jobs=n_jobs,verbose=20)(delayed(helpComputeEleIdMva)(weightsEB1,weightsEB2,weightsEE,correctedVariables,self.data[ch:ch+stride],tpC, leg2016) for ch in xrange(0,self.data.index.size,stride)))
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

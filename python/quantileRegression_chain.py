import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
import pandas as pd
import pickle as pkl
import xgboost as xgb
import gzip
import yaml
import os
import ROOT as rt
from root_pandas import read_root

from joblib import delayed, Parallel, parallel_backend, register_parallel_backend

from ..tmva.IdMVAComputer import IdMvaComputer, helpComputeIdMva
from ..tmva.eleIdMVAComputer import eleIdMvaComputer, helpComputeEleIdMva
from Corrector import Corrector, applyCorrection
#from sklearn.externals.joblib import Parallel, parallel_backend, register_parallel_backend


class quantileRegression_chain(object):
    """
    Class for performing chained quantile Regression. This only works with continious distributions.

    :param year: Year when data has been taken and MC has been produced form. Either "2016" or "2017"
    :type year: string
    :param EBEE: Barrel or Endcap. Has to be "EB" for Barrel and "EE" for endcap
    :type EBEE: string
    :param workDir: Path to the working directory. All paths are taken be relative from there, except for a *.root file in ``read_root``
    :type workDir: string
    :param varrs: List of variables to correct. Has to be ordered in the way the correction is to be preformed
    :type varrs: list
    """
    
    def __init__(self,year,EBEE,workDir,varrs):
        
        self.year = year
        self.workDir = workDir
        self.kinrho = ['probePt','probeScEta','probePhi','rho']
        if not type(varrs) is list:
            varrs=list((varrs,))
        self.vars = varrs
        self.quantiles = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
        self.backend = 'loky'
        self.EBEE = EBEE
        self.branches = ['probeScEta','probeEtaWidth','probeR9','weight','probeSigmaRR','tagChIso03','probeChIso03','probeS4','tagR9','tagPhiWidth','probePt','tagSigmaRR','probePhiWidth','probeChIso03worst','puweight','tagEleMatch','tagPhi','probeScEnergy','nvtx','probePhoIso','tagPhoIso','run','tagScEta','probeEleMatch','probeCovarianceIeIp','tagPt','rho','tagS4','tagSigmaIeIe','tagCovarianceIpIp','tagCovarianceIeIp','tagScEnergy','tagChIso03worst','probeSigmaIeIe','probePhi','mass','probeCovarianceIpIp','tagEtaWidth','probeHoE','probeFull5x5_e1x5','probeFull5x5_e5x5','probeNeutIso','probePassEleVeto']

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
        """
        Method to load a *.root dataset. Selects events in Barrel or Endcap only, depending on how class was initialized. 
        Also possible to split dataset into training and testing datasets. The dataset(s) is stored as a pandas dataframe
        in *.hd5 format.
        Arguments
        ---------
        path : string
            Path to the *.root file to be read
        tree : string
            Path to the root tree to be read with in the *.root file
        outname : string 
            Name to be used for the *.h5 file. Suffix is added automatically.
        cut : string, default ``None``
            Additional cut to apply while selecting events
        split : float, default ``None``
            Number between 0 and 1 to determine the fraction of the training sample w.r.t to the total sample size
        rndm : int, default ``12345``
            Random seed for event shuffling
        Returns
        -------
        df: pandas dataframe
            Dataframe from read *.root file
        """
        
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
        """
        Internal Method to load a dataframe from a hd5 file. Use ``loadMCDF`` to load MC dataframe of ``loadDataDF``
        to load data dataframe.
        Arguments
        ---------
        h5name : string
            Name of hd5 file to read. If not in ``workDir`` relative path from there has to be given.
        start : int, default 0
            Index of line where to start reading the df. Lines with lower indexes are not condidered
        stop : int, default -1
            Index of last line to read + 1. Set to -1 to read dataframe to the end
        rndm : int, defualt 12345
            Random seed for shuffling
        rsh : bool, default ``False``
            Set to ``True`` to reshuffle dataset while loading
        columns : list, optional
            Names of columns to be read
        Returns
        -------
        df: pandas dataframe
        """
        
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
        """
        Method to load MC dataframe. See ``_loadDf`` for Arguments
        """
        
        print 'Loading MC Dataframe from: {}/{}'.format(self.workDir,h5name)
        self.MC = self._loadDF(h5name,start,stop,rndm,rsh,columns)
        
    def loadDataDF(self,h5name,start=0,stop=-1,rndm=12345,rsh=False,columns=None):
        """
        Method to load data dataframe. See ``_loadDf`` for Arguments
        """
        
        print 'Loading data Dataframe from: {}/{}'.format(self.workDir,h5name)
        self.data = self._loadDF(h5name,start,stop,rndm,rsh,columns)

    def trainOnData(self,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):
        """
        Method to train quantile regression BDTs on data. See ``_trainQuantiles`` for Arguments
        """
        
        self._trainQuantiles('data',var=var,maxDepth=maxDepth,minLeaf=minLeaf,weightsDir=weightsDir)
        
    def trainOnMC(self,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):
        """
        Method to train quantile regression BDTs on MC. See ``_trainQuantiles`` for Arguments
        """
        
        self._trainQuantiles('mc',var=var,maxDepth=maxDepth,minLeaf=minLeaf,weightsDir=weightsDir)

    def _trainQuantiles(self,key,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):
        """
        Internal method to train BDTs for quantile morphing. All trees for one variable are trained in parallel. 
        An ipcluster can be used a ipyparallel backend. Call ``register_parallel_backend`` to set this up.
        If no ipcluster is set up, loky backend will be used and spawn 21 processes locally. The trained BDTs will be
        pickled, zipped and stored in ``weightsDir```
        Arguments
        ---------
        key : string
            Specify if training is run on data or MC. Hast to be "data" or "mc"
        var : string
            Name of the varible to be trained on
        maxDepth : int, default 5
            max_depth of trees that will be trained
        minLeaf : int, default 500
            Sets min_samples_leaf and min_samples_split in tree training
        weightsDir : str, default "/weights_qRC"
            Directory the weight files will be saved to. Relative to ``workDir`` 
        """
        
        if var not in self.vars+['{}_shift'.format(x) for x in self.vars]:
            raise ValueError('{} has to be one of {}'.format(var, self.vars))
        
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

        print 'Training quantile regression on {} for {} with features {}'.format(key,var,features)

        with parallel_backend(self.backend):
            Parallel(n_jobs=len(self.quantiles),verbose=20)(delayed(trainClf)(q,maxDepth,minLeaf,X,Y,save=True,outDir='{}/{}'.format(self.workDir,weightsDir),name='{}_weights_{}_{}_{}'.format(name_key,self.EBEE,var,str(q).replace('.','p')),X_names=features,Y_name=var) for q in self.quantiles)

        
    def correctY(self, var, n_jobs=1, diz=False):
        """
        Medthod to apply correction for  one variable in MC. BDTs for data and MC have to be loaded first, 
        using ``loadClfs``.
        Make sure to load the right files, variable names are not checked. If ``n_jobs`` is bigger than 1,
        correction will be run in parallel, with ipcluster backend if available, otherwise loky
        Arguments
        ---------
        var : string
            Name of varible to be corrected
        n_jobs : int, default 1
            Number of parallel processes to use
        diz : bool, defautl ``False``
            Specify if variable to be corrected is discontinuous. Only for ``quantileRegression_chain_disc``
        """
        
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
        """
        Method to train one BDT for final application of correction. Variable needs to be corrected
        using ``correctY`` before. The training target is the scaled difference between corrected and 
        uncorrected values of the variable. Therefore a scaler is saved along with the trained BDT itself,
        both pickled and zipped. The parameters of the tree will be taken from ``finalRegression_settings.yaml``
        if available in ``weightsDir``, otherwise default settings are used.
        Arguments
        ---------
        var : string
            Name of the variable to train for
        weightsDir : string
            Directory of the multiple BDT weights, scaler trained single BDT will also be saved here.
        diz : bool, default ``False``
            Specifies if varible to train for is discontinuous. Only used by ``quantileRegression_chain_disc``
        n_jobs : int, default 1
            Number of threads to be used for the training with XGBoost. This happens interally in XGBoost. 
            An ipyparallel backend will not beused, even if set up.
        """
        
        robSca = RobustScaler()
        features = self.kinrho + self.vars
        target = '{}_corr_diff_scale'.format(var)

        if diz:
            querystr = '{0}!=0 and {0}_corr!=0'.format(var)
        else:
            querystr = '{0}=={0}'.format(var)

        df = self.MC.query(querystr)

        df['{}_corr_diff_scale'.format(var)] = robSca.fit_transform(np.array(df['{}_corr'.format(var)] - df[var]).reshape(-1,1))
        pkl.dump(robSca,gzip.open('{}/{}/scaler_mc_{}_{}_corr_diff.pkl'.format(self.workDir,weightsDir,self.EBEE,var),'wb'),protocol=pkl.HIGHEST_PROTOCOL)

        X = df.loc[:,features].values
        Y = df[target].values

        try:
            settings = yaml.load(open('{}/{}/finalRegression_settings.yaml'.format(self.workDir,weightsDir)))
            clf = xgb.XGBRegressor(base_score=0.,n_jobs=n_jobs,**settings[var])
            print('Custom settings loaded')
        except (IOError,KeyError):
            print('No custom settings found, training with standard settings')
            clf = xgb.XGBRegressor(n_estimators=1000, max_depth=10, gamma=0, base_score=0.,n_jobs=n_jobs)

        print('Training final Regression for {} with features {}. Classifier:{}'.format(target,features,clf))
        clf.fit(X,Y)

        name = 'weights_finalRegressor_{}_{}'.format(self.EBEE,var)
        print 'Saving final regrssion trained with features {} for {} to {}/{}.pkl'.format(features,'{}_corr_diff_scale'.format(var),weightsDir,name)
        dic = {'clf': clf, 'X': features, 'Y': '{}_corr_diff_scale'.format(var)}
        pkl.dump(dic,gzip.open('{}/{}/{}.pkl'.format(self.workDir,weightsDir,name),'wb'),protocol=pkl.HIGHEST_PROTOCOL)
        
    def loadFinalRegression(self,var,weightsDir):
        """
        Method to load a final regressor. 
        Parameters
        ----------
        var : string
            Name of the variable the regressor is loaded for
        weightsDir: string
            Directory the regressor is loaded from, relaitve to ``workDir``
        """
        
        self.finalReg = self.load_clf_safe(var,weightsDir,'weights_finalRegressor_{}_{}.pkl'.format(self.EBEE,var),self.kinrho+self.vars,'{}_corr_diff_scale'.format(var))

    def loadScaler(self,var,weightsDir):
        """
        Method to load a scaler for final regressor
        Parameters
        ----------
        var : string
            Name of the variable the scaler is loaded for
        weightsDir: string
            Directory the scaler is loaded from, relaitve to ``workDir``
        """
        
        self.scaler = pkl.load(gzip.open('{}/{}/scaler_mc_{}_{}_corr_diff.pkl'.format(self.workDir,weightsDir,self.EBEE,var)))

    def applyFinalRegression(self,var,diz=False):
        """
        Method to apply correction using the final single BDT trained with ``trainFinalRegression``. 
        Parameters
        ----------
        var : string
            Name of the variable to be corrected
        diz : bool, default ``False``
            Specifies if variable to be corrected is discontinuous. Only used by ``quantileRegression_chain_disc``
        """
        var_raw = var[:var.find('_')] if '_' in var else var
        features = self.kinrho + self.vars
        if diz:
            X = self.MC.loc[self.MC[var] != 0.,features].values
            self.MC.loc[self.MC[var] != 0.,'{}_corr_1Reg'.format(var_raw)] = self.MC.loc[self.MC[var] != 0.,var] + self.scaler.inverse_transform(self.finalReg.predict(X).reshape(-1,1)).ravel()
            self.MC.loc[self.MC[var] == 0.,'{}_corr_1Reg'.format(var_raw)] = 0.
        else:
            X = self.MC.loc[:,features].values
            self.MC['{}_corr_1Reg'.format(var)] = self.MC[var] + self.scaler.inverse_transform(self.finalReg.predict(X).reshape(-1,1)).ravel()
        
    def trainAllMC(self,weightsDir,n_jobs=1):
        """
        Method to train all BDTs for MC. Multiple BDTs per variable are trained and applied, such that the
        BDTs for the next variable can be trained with the other corrected variables as inputs.
        Arguments
        ---------
        weightsDir : string
            Directory where the weight files will be stored. Data weights have to be in there.
        n_jobs: int
            Number of jobs used for applying the previously trained BDTs. Uses ipcluster if set up.
        """
        
        for var in self.vars:
            try:
                self.loadClfs(var,weightsDir)
            except IOError:
                self.trainOnMC(var,weightsDir=weightsDir)
                self.loadClfs(var,weightsDir)

            self.correctY(var,n_jobs=n_jobs)
            
    def loadClfs(self, var, weightsDir):
        """
        Method to load mulitple BDTs for data and MC simultaneously. See ``load_clf_safe`` for details.
        Arguments
        ---------
        var : string
            Name of variable regressors are loaded for
        weightsDir : string
            Directory where weights are stored. Relative to workDir
        """
        
        self.clfs_mc = [self.load_clf_safe(var, weightsDir, 'mc_weights_{}_{}_{}.pkl'.format(self.EBEE,var,str(q).replace('.','p'))) for q in self.quantiles]
        self.clfs_d = [self.load_clf_safe(var, weightsDir,'data_weights_{}_{}_{}.pkl'.format(self.EBEE,var,str(q).replace('.','p'))) for q in self.quantiles]
        
    def load_clf_safe(self,var,weightsDir,name,X_name=None,Y_name=None):
        """
        General method to load regressor stored in pkl file. The file has to contain a dictionay
        with input variables of the regressor and the variable it was trained for. While loading this
        is compared to ``X_name`` (input variables) and ``Y_name`` (variable trained for). If something
        does not match, an exception is thrown
        Arguments
        ---------
        var : string
            Name of varibles the regressor is loaded for
        weightDir : string
            Directory where the weight file is stored. Relative to workDir
        name : string
            Filename of the weight file
        X_name : string, default ``None``
            List of variables used as inputs for the BDT training. Set automatically 
            if ``None``.
        Y_name : string, default ``None``
            Name of variable the BDT is trained for. Is set to ``var`` if it is ``None``
        """
        
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
        """
        Method to get the value of the conditional CDF of one variable for one event. 
        Needed for ``quantileRegression_chain_disc``
        Arguments
        ---------
        df : pandas dataframe
            Dataframe to get the values for. It will be stored in there
        clfs : list
            List of regressors for variable
        features : list
            List of features for the regressors
        var : string
            Name of variable to get quantile values for
        Returns
        -------
        numpy array : With cdf values for each event. Shape : ``[n_evts,1]``
        """
        
        qtls_names = ['q{}_{}'.format(str(self.quantiles[i]).replace('0.','p'),var) for i in range(len(self.quantiles))]
        X = df.loc[:,features]
        if not all(qtls_names) in df.columns:
            mcqtls = [clf.predict(X) for clf in clfs]
            for i in range(len(self.quantiles)):
                df[qtls_names[i]] = mcqtls[i]
                
        return df.loc[:,[var] + qtls_names].apply(self._getCDFval,1,raw=True)
        
    def _getCDFval(self,row):
        """
        Helper method to get cdf value through linear interpolation
        between predicted quantile values.
        Arguments
        ---------
        row : numpy arraw
            Array with value of variable and predicted quantile values
        Returns
        -------
        double : 
        """
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
        """
        Method to evaluate several versions of the photon IdMVA. Calls ``computeIdMva``
        For Arguments not listed here, see ``computeIdMva``
        mvas : list of 3-tuples
            The first entry of the tuple defines the name that is used to store the result in the
            data/MC dataframe, the second entry specifies the kind of correction that was applied
            the third entry is a list of corrected variables to be used for the calculation of this
            version of the IdMVA
        weights : 2-tuple, strings
            The first entry of the tuple is the path to the photon IDMVA weight file for EB, 
            the second entry is the path to the photon IdMVA weight file for EE
        """
        
        weightsEB,weightsEE = weights
        for name,tpC,correctedVariables in mvas:
            self.computeIdMva(name,weightsEB,weightsEE,key,correctedVariables,tpC,leg2016,n_jobs)

    def computeIdMva(self,name,weightsEB,weightsEE,key,correctedVariables,tpC,leg2016,n_jobs):
        """
        Method to compute value of photon IdMVA for every event in data and MC
        Arguments
        ---------
        name : string
            name of the version of the photon IdMVA. Is used as column name in the dataframe,
            for which the photon IdMVA is computed
        weightsEB : string
            path to the photon IDMVA weight file for EB
        weightsEE : string
            path to the photon IDMVA weight file for EE
        key : string
            Specifies if the MVA is computed for data or mc. Has to be "data" or "mc"
        correctedVariables : list
            List of variables for which the corrected version is used for the photon IdMVA calculation
        tpC : string
            The kind of correction that that will be used for the calculation
        leg2016 : bool, default ``False``
            Specifies if photon IdMVA is computed for data/MC from year 2016
        n_jobs : int, default 1
            Number of parallel jobs to be used for the computation of the photon IdMVA
        """
        
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
        """
        Method to evaluate several version of the electron IdMVA. Calls ``computeEleIdMva``. 
        For Arguments see ``computeIdMvas``
        """
        
        weightsEB1,weightsEB2,weightsEE = weights
        for name,tpC,correctedVariables in mvas:
            self.computeEleIdMva(name,weightsEB1,weightsEB2,weightsEE,key,correctedVariables,tpC,leg2016,n_jobs)

    def computeEleIdMva(self,name,weightsEB1,weightsEB2,weightsEE,key,correctedVariables,tpC,leg2016,n_jobs):
        """
        Method to compute the electron IdMVA. Uses ``helpComputeEleIdMva`` from ``..tmva.eleIdMVAComputer``.
        For Arguments see ``computeIdMva``
        """
        
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


    def setupJoblib(self,ipp_profile='default',cluster_id=None):
        """
        Method to set ipyparallel backend to a running ipcluster
        Arguments
        ---------
        ipp_profile : string
            Name of ipcluster profile for the started ipcluster that will be set up
        """
        
        import ipyparallel as ipp
        from ipyparallel.joblib import IPythonParallelBackend
        global joblib_rc,joblib_view,joblib_be
        joblib_rc = ipp.Client(profile=ipp_profile, cluster_id=cluster_id)
        joblib_view = joblib_rc.load_balanced_view()
        joblib_be = IPythonParallelBackend(view=joblib_view)
        register_parallel_backend('ipyparallel',lambda: joblib_be, make_default=True)

        self.backend = 'ipyparallel'


def trainClf(alpha,maxDepth,minLeaf,X,Y,save=False,outDir=None,name=None,X_names=None,Y_name=None):
    """
    Method to train a BDT using the quantile regression
    Arguments
    ---------
    alpha : float
        Value of the cdf to train form
    maxDepth : int
        max_depth set to this value for the training of the regressor
    minLeaf : int
        min_samples_leaf and min_samples_split set to this value for training
    X : numpy array
        Array of input values for the training, shape ``[n_evts, n_input_variables]``
    Y : numpy array
        Target array for training, shape ``[n_evts,1]``
    save : bool, default False
        Specifies if trained regressor is to be saved
    outDir : string
        path to the directory where trained classifier is stored. Only used if ``save`` is ``True``
    X_names : list
        List of variable names of the input variables, to be written in the stored file. 
        Only used if ``save`` is ``True``
    Y_name : string
        Name of the target variable, to be written in the stored file. Only used if ``save`` is ``True``
    """
    
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

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import pickle as pkl
import gzip
import os
from joblib import delayed, Parallel, parallel_backend, register_parallel_backend
import ROOT as rt
#from sklearn.externals.joblib import Parallel, parallel_backend, register_parallel_backend


class quantileRegression_chain:

    def __init__(self,year,EBEE,workDir):

        self.year = year
        self.workDir = workDir
        if year == '2017':
            self.ShowerShapes = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
        elif year == '2016':
            self.ShowerShapes = ['probeCovarianceIetaIphi','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
        self.kinrho = ['probePt','probeScEta','probePhi','rho']
        self.quantiles = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
        self.backend = 'loky'
        self.EBEE = EBEE


    def loadDataDF(self, h5name, start=0, stop=-1, rndm=12345, rsh=False, columns=None):
        
        try:
            print 'Loading Data Dataframe from: ', self.workDir+'/'+h5name
            if rsh:
                df = pd.read_hdf(self.workDir+'/'+h5name, 'df', columns=columns)
            else:
                df = pd.read_hdf(self.workDir+'/'+h5name, 'df', columns=columns, start=start, stop=stop)
        except IOError:
            print 'h5 file does not exist'
        
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

        self.data = df

    def loadMCDF(self,h5name,start,stop,rndm=12345,rsh=False,columns=None):
        
        try:
            print 'Loading Monte-Carlo Dataframe from: ', self.workDir+'/'+h5name
            df = pd.read_hdf(self.workDir+'/'+h5name, 'df', columns=columns)
        except IOError:
            print 'h5 file does not exist'
        
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

        self.MC = df

    def trainOnData(self,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):
        
        if var not in self.ShowerShapes:
            raise ValueError('{} has to be one of {}'.format(var, self.ShowerShapes))
        
        features = self.kinrho + self.ShowerShapes[:self.ShowerShapes.index(var)]
        X = self.data.loc[:,features]
        Y = self.data[var]

        print 'Training regressors on data'
        with parallel_backend(self.backend):
            Parallel(n_jobs=len(self.quantiles),verbose=20)(delayed(trainClf)(q,maxDepth,minLeaf,X,Y,save=True,outDir='{}/{}'.format(self.workDir,weightsDir),name='data_weights_{}_{}_{}'.format(self.EBEE,var,str(q).replace('.','p')),X_names=features,Y_name=var) for q in self.quantiles)
        
        # print 'Saving regressors trained on data'
        # dic = {'clf': clf, 'X': features, 'Y': var}
        # pkl.dump(dic,gzip.open('{}/{}/data_weights_{}_{}.pkl'.format(self.workDir,weightsDir,var,alpha),'wb'),protocol=pkl.HIGHEST_PROTOCOL)
    
    def trainOnMC(self,var,maxDepth=5,minLeaf=500,weightsDir='/weights_qRC'):
        
        if var not in self.ShowerShapes:
            raise ValueError('{} has to be one of {}'.format(var, ShowerShapes))
        
        features = self.kinrho + ['{}_corr'.format(x) for x in self.ShowerShapes[:self.ShowerShapes.index(var)]]
        X = self.MC.loc[:,features]
        Y = self.MC[var]

        print 'Training regressor on MC'
        with parallel_backend(self.backend):
            Parallel(n_jobs=len(self.quantiles),verbose=20)(delayed(trainClf)(q,maxDepth,minLeaf,X,Y,save=True,outDir='{}/{}'.format(self.workDir,weightsDir),name='mc_weights_{}_{}_{}'.format(self.EBEE,var,str(q).replace('.','p')),X_names=features,Y_name=var) for q in self.quantiles)

        # print 'Saving regressor trained on mc'
        # clfs_mc.append(self.ShowerShapes)
        # pkl.dump(clfs_mc,gzip.open('{}/{}/mc_clfs_{}.pkl'.format(self.workDir,weightsDir,alpha),'wb'),protocol=pkl.HIGHEST_PROTOCOL)
        
    def correctY(self, var, n_jobs=1, store=True):
        
        features = self.kinrho + ['{}_corr'.format(x) for x in self.ShowerShapes[:self.ShowerShapes.index(var)]]
        X = self.MC.loc[:,features]
        Y = self.MC[var]
        
        if X.isnull().values.any():
            # print 'Correct {} first !'.format(self.ShowerShapes[:self.ShowerShapes.index(var)])
            raise KeyError('Correct {} first!'.format(self.ShowerShapes[:self.ShowerShapes.index(var)]))

        print "Features: X = ", features, " target y = ", var
        
        Y = Y.values.reshape(-1,1)
        Z = np.hstack([X,Y])

        # clf_mc = [self.clfs_mc[i][var] for i in range(len(self.quantiles))]
        # clf_d = [self.clfs_d[i][var] for i in range(len(self.quantiles))]

        with parallel_backend(self.backend):
            Ycorr = np.concatenate(Parallel(n_jobs=n_jobs,verbose=20)(delayed(applyCorrection)(self.clfs_mc,self.clfs_d,ch[:,:-1],ch[:,-1]) for ch in np.array_split(Z,n_jobs) ) )

        if store:
            self.MC['{}_corr'.format(var)] = Ycorr

    def trainAllMC(self,weightsDir):
        
        for var in self.ShowerShapes:
            self.trainOnMC(var,weightsDir=weightsDir)
            self.loadClfs(var,weightsDir)
            self.correctY(var,n_jobs=20)
            
    def loadClfs(self, var, weightsDir):
        
        self.clfs_mc = [self.load_clf_safe('mc', weightsDir, var, q) for q in self.quantiles]
        self.clfs_d = [self.load_clf_safe('data', weightsDir, var, q) for q in self.quantiles]
        
    def load_clf_safe(self,key,weightsDir,var,q):
        
        clf = pkl.load(gzip.open('{}/{}/{}_weights_{}_{}_{}.pkl'.format(self.workDir,weightsDir,key,self.EBEE,var,str(q).replace('.','p'))))
        if key == 'mc':
            if clf['X'] != self.kinrho + ['{}_corr'.format(x) for x in self.ShowerShapes[:self.ShowerShapes.index(var)]] or clf['Y'] != var:
                raise ValueError('{}/{}/{}_weights_{}_{}_{}.pkl was not trained with the right order of ShowerShapes!'.format(self.workDir,weightsDir,key,self.EBEE,var,str(q).replace('.','p')))
            else:
                return clf['clf']

        if key == 'data':
            if clf['X'] != self.kinrho + ['{}'.format(x) for x in self.ShowerShapes[:self.ShowerShapes.index(var)]] or clf['Y'] != var:
                raise ValueError('{}/{}/{}_weights_{}_{}_{}.pkl was not trained with the right order of ShowerShapes!'.format(self.workDir,weightsDir,key,self.EBEE,var,str(q).replace('.','p')))
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
                Y = np.concatenate(Parallel(n_jobs=n_jobs,verbose=20)(delayed(computeIdMva)(weightsEB,weightsEE,correctedVariables,self.MC[ch:ch+stride],tpC, leg2016) for ch in xrange(0,self.MC.index.size,stride)))
            self.MC[name] = Y
        elif key == 'data':
            with parallel_backend(self.backend):
                Y = np.concatenate(Parallel(n_jobs=n_jobs,verbose=20)(delayed(computeIdMva)(weightsEB,weightsEE,correctedVariables,self.data[ch:ch+stride],tpC, leg2016) for ch in xrange(0,self.data.index.size,stride)))
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

class Corrector:
   
   # store regressors
   def __init__(self,mcclf,dataclf,X,Y,diz=False):
      self.diz=diz #Flag for distribution with discrete 0, i.e. Isolation
      self.mcqtls   = np.array([clf.predict(X) for clf in mcclf])
      self.dataqtls = np.array([clf.predict(X) for clf in dataclf])

      self.Y = Y
      
   # correction is actually done here
   def correctEvent(self,iev):

      mcqtls = self.mcqtls[:,iev]
      dataqtls = self.dataqtls[:,iev]
      Y = self.Y[iev]
      
      if self.diz and Y == 0.:
         return 0.

      qmc =0
      
      while qmc < len(mcqtls): # while + if, to avoid bumping the range
         if mcqtls[qmc] < Y:
            qmc+=1
         else:
            break

      if qmc == 0:
         qmc_low,qdata_low   = 0,0                              # all shower shapes have a lower bound at 0
         qmc_high,qdata_high = mcqtls[qmc],dataqtls[qmc]
      elif qmc < len(mcqtls):
         qmc_low,qdata_low   = mcqtls[qmc-1],dataqtls[qmc-1]
         qmc_high,qdata_high = mcqtls[qmc],dataqtls[qmc]
      else:
         qmc_low,qdata_low   = mcqtls[qmc-1],dataqtls[qmc-1]
         qmc_high,qdata_high = mcqtls[len(mcqtls)-1]*1.2,dataqtls[len(dataqtls)-1]*1.2
         # to set the value for the highest quantile 20% higher
                                                                       
      return (qdata_high-qdata_low)/(qmc_high-qmc_low) * (Y - qmc_low) + qdata_low

   def __call__(self):
      return np.array([ self.correctEvent(iev) for iev in xrange(self.Y.size) ]).ravel()

def applyCorrection(mcclf,dataclf,X,Y,diz=False):
   return Corrector(mcclf,dataclf,X,Y,diz)()

def clf_convert_to_dic(clfs):
    
    d_clfs = {}
    for i in range(len(clfs)-1):
        d_clfs[clfs[-1][i]] = clfs[i]

    return d_clfs
    
class IdMvaComputer:

   def __init__(self,weightsEB,weightsEE,correct=[],tpC='qr',leg2016=False):
      rt.gROOT.LoadMacro("/mnt/t3nfs01/data01/shome/threiten/QReg/dataMC-1/MTR/phoIDMVAonthefly.C")
      
      self.rhoSubtraction = False
      # if type(correct) == dict:
      #    self.rhoSubtraction = correct["rhoSubtraction"]
      #    correct = correct["correct"]
         
      
      self.tpC = tpC
      self.leg2016 = leg2016
      self.X = rt.phoIDInput()
      self.readerEB = rt.bookReadersEB(weightsEB, self.X)
      
      self.readerEE = rt.bookReadersEE(weightsEE, self.X, self.rhoSubtraction, self.leg2016)
      
      # print ("IdMvaComputer.__init__")
      if leg2016:
         columns = ["probeScEnergy","probeScEta","rho","probeR9","probeSigmaIeIe","probePhiWidth","probeEtaWidth","probeCovarianceIetaIphi","probeS4","probePhoIso","probeChIso03","probeChIso03worst","probeSigmaRR","probeScPreshowerEnergy","probePt"]
      else:
         columns = ["probeScEnergy","probeScEta","rho","probeR9","probeSigmaIeIe","probePhiWidth","probeEtaWidth","probeCovarianceIeIp","probeS4","probePhoIso","probeChIso03","probeChIso03worst","probeSigmaRR","probeScPreshowerEnergy","probePt"]

      if self.rhoSubtraction:
         self.effareas = np.array([[0.0000, 0.1210],   
                                   [1.0000, 0.1107],
                                   [1.4790, 0.0699],
                                   [2.0000, 0.1056],
                                   [2.2000, 0.1457],
                                   [2.3000, 0.1719],
                                   [2.4000, 0.1998],
         ])
         
      # make list of input columns
      if self.tpC=="qr":
         print "Using variables corrected by quantile regression"
         self.columns = map(lambda x: x+"_corr" if x in correct else x, columns)
         print self.columns

      elif self.tpC=="old":
         print "Using variables corrected by old method"
         self.columns = map(lambda x: x+"_old_corr" if x in correct else x, columns)
         print self.columns

      elif self.tpC=="data":
         print "Using uncorrected variables"
         self.columns = columns
         print self.columns

      elif self.tpC=="n-1qr":
         print "Using variables corrected by N-1 quantile regression"
         self.columns = map(lambda x: x+"_corr_corrn-1" if x in correct else x, columns)
         print self.columns

      elif self.tpC=="n-1qrnc":
         print "Using variables corrected by N-1 nc quantile regression"
         self.columns = map(lambda x: x+"_corrn-1" if x in correct else x, columns)
         print self.columns

      elif self.tpC=="I2qr":
         print "Using variables corrected by I2 quantile regression"
         self.columns = map(lambda x: x+"_corr_corrn-1_corr" if x in correct else x, columns)
         print self.columns
      
      elif self.tpC=="I2n-1qr":
         print "Using variables corrected by I2 N-1 quantile regression"
         self.columns = map(lambda x: x+"_corr_corrn-1_corr_corrn-1" if x in correct else x, columns)
         print self.columns
      
   def __call__(self,X):

      # make sure of order of the input columns and convert to a numpy array
      Xvals = X[self.columns ].values
      #print self.columns
      #print Xvals[0]
 
      return np.apply_along_axis( self.predict, 1, Xvals ).ravel()
      
   def predict(self,row):
      return self.predictEB(row) if np.abs(row[1]) < 1.5 else self.predictEE(row)
      # return self.predictEB(row)
      
   def predictEB(self,row):
      # use numeric indexes to speed up
      #print ("IdMvaComputer.predictEB")
      self.X.phoIdMva_SCRawE_          = row[0]
      self.X.phoIdMva_ScEta_           = row[1]
      self.X.phoIdMva_rho_             = row[2]
      self.X.phoIdMva_R9_              = row[3]
      self.X.phoIdMva_covIEtaIEta_     = row[4] # this is really sieie
      self.X.phoIdMva_PhiWidth_        = row[5]
      self.X.phoIdMva_EtaWidth_        = row[6]
      self.X.phoIdMva_covIEtaIPhi_     = row[7]
      self.X.phoIdMva_S4_              = row[8]
      self.X.phoIdMva_pfPhoIso03_      = row[9]
      self.X.phoIdMva_pfChgIso03_      = row[10]
      self.X.phoIdMva_pfChgIso03worst_ = row[11]
      return self.readerEB.EvaluateMVA("BDT")

   def effArea(self,eta):
      ibin = min(self.effareas.shape[0]-1,bisect.bisect_left(self.effareas[:,0],eta))
      return self.effareas[ibin,1]
   
   def predictEE(self,row):
      #print "IdMvaComputer.predictEE"
      self.X.phoIdMva_SCRawE_          = row[0]
      self.X.phoIdMva_ScEta_           = row[1]
      self.X.phoIdMva_rho_             = row[2]
      self.X.phoIdMva_R9_              = row[3]
      self.X.phoIdMva_covIEtaIEta_     = row[4] # this is really sieie
      self.X.phoIdMva_PhiWidth_        = row[5]
      self.X.phoIdMva_EtaWidth_        = row[6]
      self.X.phoIdMva_covIEtaIPhi_     = row[7]
      self.X.phoIdMva_S4_              = row[8]
      self.X.phoIdMva_pfPhoIso03_      = row[9]
      if self.rhoSubtraction: self.X.phoIdMva_pfPhoIso03_ = max(2.5, self.X.phoIdMva_pfPhoIso03_ - self.effArea( np.abs(self.X.phoIdMva_ScEta_))*self.X.phoIdMva_rho_ - 0.0034*row[14] )
      self.X.phoIdMva_pfChgIso03_      = row[10]
      self.X.phoIdMva_pfChgIso03worst_ = row[11]
      self.X.phoIdMva_ESEffSigmaRR_    = row[12]
      esEn                             = row[13]
      ScEn                             = row[0]
      self.X.phoIdMva_esEnovSCRawEn_ = esEn/ScEn
      return self.readerEE.EvaluateMVA("BDT")


def computeIdMva(weightsEB,weightsEE,correct,X,tpC,leg2016):
   return IdMvaComputer(weightsEB,weightsEE,correct,tpC,leg2016)(X)

import numpy as np
import ROOT as rt
from joblib import delayed, Parallel

class qRC_Computer:

   def __init__(self,weights,scaler,leg2016=False):#,weightsEE,correct=[],tpC='qr'):
      rt.gROOT.LoadMacro("/mnt/t3nfs01/data01/shome/threiten/QReg/qRC/qRC_xmlReader_PhoIso.C")
      
      self.X = rt.qRC_Input()
      self.readerEB = rt.bookReaders(weights, self.X)
      self.scaler = scaler

      if leg2016:
          pass
###Not working          self.columns = ["probePt","probeScEta","probePhi","rho","probePhiWidth","probeCovarianceIetaIphi","probeS4","probeR9","probeSigmaIeIe","probeEtaWidth"]
      else:
          self.columns = ["probePt","probeScEta","probePhi","rho","probePhoIso"]

   def predict(self,row):
       
       self.X.qRC_Input_pt_=row[0]
       self.X.qRC_Input_ScEta_=row[1]
       self.X.qRC_Input_Phi_=row[2]
       self.X.qRC_Input_rho_=row[3]
       self.X.qRC_Input_phoIso_=row[4]

       reg = self.readerEB.EvaluateRegression("BDT")[0]
       # print reg
       return reg

   def __call__(self,X):
       Xvals = X[self.columns].values
       return np.apply_along_axis( self.predict, 1, Xvals ).reshape(-1,1)


def helpCompute_qRC(weights,X,scaler,leg2016):
    return qRC_Computer(weights,scaler,leg2016)(X)

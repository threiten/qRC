import numpy as np
import ROOT as rt
from joblib import delayed, Parallel

class qRC_Computer:

   def __init__(self,weights,leg2016=False):#,weightsEE,correct=[],tpC='qr'):
      rt.gROOT.LoadMacro("/mnt/t3nfs01/data01/shome/threiten/QReg/qRC/qRC_xmlReader.C")
      
      self.X = rt.qRC_Input()
      self.readerEB = rt.bookReaders(weights, self.X)

      if leg2016:
          pass
###Not working          self.columns = ["probePt","probeScEta","probePhi","rho","probePhiWidth","probeCovarianceIetaIphi","probeS4","probeR9","probeSigmaIeIe","probeEtaWidth"]
      else:
          self.columns = ["probePt","probeScEta","probePhi","rho","probeCovarianceIeIp","probeS4","probeR9","probePhiWidth","probeSigmaIeIe","probeEtaWidth"]

   def predict(self,row):
       
       self.X.qRC_Input_pt_=row[0]
       self.X.qRC_Input_ScEta_=row[1]
       self.X.qRC_Input_Phi_=row[2]
       self.X.qRC_Input_rho_=row[3]
       self.X.qRC_Input_covaIEtaIPhi_=row[4]
       self.X.qRC_Input_S4_=row[5]
       self.X.qRC_Input_R9_=row[6]
       self.X.qRC_Input_phiWidth_=row[7]
       self.X.qRC_Input_sigmaIEtaIEta_=row[8]
       self.X.qRC_Input_etaWidth_=row[9]
       self.predict = self.readerEB.EvaluateRegression("BDTG")
       return self.predict

   def __call__(self,X):
       Xvals = X[self.columns].values
       return np.apply_along_axis( self.predict, 1, Xvals ).ravel()


def helpCompute_qRC(weights,X,leg2016):
    return qRC_Computer(weights,leg2016)(X)

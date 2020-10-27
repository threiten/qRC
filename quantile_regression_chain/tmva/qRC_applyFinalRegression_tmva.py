import pandas as pd
import numpy as np
import ROOT as rt

class computeCorrection_tmva:

    def __init__(self,scl_center,scl_iqr,weights,leg2016=False):
    
        rt.gROOT.LoadMacro("/work/threiten/QReg/qRC/tmva/qRC_xmlReader.C")
      
        self.X = rt.qRC_Input()
        self.readerEB = rt.bookReaders(weights, self.X)
        self.scl_center=scl_center
        self.scl_iqr=scl_iqr
        
    def __call__(self,row):

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
        predict = self.readerEB.EvaluateRegression("BDTG")
        return self.scl_iqr*predict + self.scl_center

def applyFinalRegression_tmva(var,df,scaler,weights,leg2016):
    
    row=df
    correction = np.apply_along_axis(computeCorrection_tmva(scaler.center_[0],scaler.scale_[0],weights,leg2016),1,row)
    
    return correction
    

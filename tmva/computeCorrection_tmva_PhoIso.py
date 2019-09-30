import pandas as pd
import numpy as np
import ROOT as rt

class computeCorrection_tmva_Iso:

    def __init__(self,scl_center,scl_iqr,weightsFinalReg,weightsFinalTailReg,weightsDataClf,weightsMcClf,leg2016=False):
    
        rt.gROOT.LoadMacro("/t3home/threiten/python/qRC/tmva/qRC_xmlReader_PhoIso.C")
      
        self.X = rt.qRC_Input_Iso()
        self.readerFinalReg = rt.bookReaderFinalReg(weightsFinalReg, self.X)
        self.readerTailReg = rt.bookReaderTailReg(weightsFinalTailReg, self.X)
        self.readerDataClf = rt.bookReaderpotClf(weightsDataClf, self.X)
        self.readerMcClf = rt.bookReaderpotClf(weightsMcClf,self.X)
        self.scl_center=scl_center
        self.scl_iqr=scl_iqr

    def shiftY(self):
        
        r=np.random.uniform()

        #TMVA_pred=2*xgboost_pred-1, value stored in trees from xgboost is xgboost_pred-0.5 (=base_score)
        pPeak_data=((self.readerDataClf.EvaluateMVA("potClf")+1)/2)+0.5
        pPeak_mc=((self.readerMcClf.EvaluateMVA("potClf")+1)/2)+0.5

        drats=[]
        drats.append(((1-pPeak_data)-(1-pPeak_mc))/pPeak_mc)
        drats.append(((pPeak_data)-(pPeak_mc))/(1-pPeak_mc))
        
        if self.X.qRC_Input_phoIso_ == 0. and (1-pPeak_data)>(1-pPeak_mc) and r<=drats[0]:
            shifted = self.p2t()
        elif self.X.qRC_Input_phoIso_ > 0. and pPeak_data>pPeak_mc and r<=drats[1]:
            shifted = 0.
        else:
            shifted = self.X.qRC_Input_phoIso_

        return shifted

    def p2t(self):
        
        return self.readerTailReg.EvaluateRegression("tailReg")[0]
        
    def __call__(self,row):

        self.X.qRC_Input_pt_ = row[0]
        self.X.qRC_Input_ScEta_ = row[1]
        self.X.qRC_Input_Phi_ = row[2]
        self.X.qRC_Input_rho_ = row[3]
        self.X.qRC_Input_phoIso_ = row[4]
        # Conditional CDF from quantileRegression with first quantile 0.01 and last quantile 0.99
        self.X.qRC_Input_rand01_ = np.random.uniform(0.01,0.99)

        self.X.qRC_Input_phoIso_ = float(self.shiftY())
        if self.X.qRC_Input_phoIso_ == 0.:
            return self.X.qRC_Input_phoIso_
        elif self.X.qRC_Input_phoIso_ > 0.:
            return self.X.qRC_Input_phoIso_ + self.scl_iqr*self.readerFinalReg.EvaluateRegression("finalReg")[0]+self.scl_center

def applyFinalRegressionsIso_tmva(var,df,scaler,weightsFinalReg,weightsFinalTailReg,weightsDataClf,weightsMcClf,leg2016):
    
    # columns = ["probePt","probeScEta","probePhi","rho","probePhoIso"]
    # row=df[columns].values
    row=df
    correction = np.apply_along_axis(computeCorrection_tmva_Iso(scaler.center_[0],scaler.scale_[0],weightsFinalReg,weightsFinalTailReg,weightsDataClf,weightsMcClf,leg2016),1,row)
    return correction
    # df['{}_corr_tmva'.format(var)] = correction.ravel()

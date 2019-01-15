import pandas as pd
import numpy as np
import ROOT as rt

class computeCorrection_tmva_Iso:

    def __init__(self,scl_center,scl_iqr,weightsFinalReg,weightsTailReg,weightsDataClf,weightsMcClf,leg2016=False):
    
        rt.gROOT.LoadMacro("/mnt/t3nfs01/data01/shome/threiten/QReg/qRC/qRC_xmlReader_PhoIso.C")
      
        self.X = rt.qRC_Input()
        self.readerFinalReg = rt.bookReaderFinalReg(weightsFinalReg, self.X)
        self.readerTailReg = rt.bookReaderTailReg(weightsTailReg, self.X)
        self.readerDataClf = rt.bookReaderDataClf(weightsDataClf, self.X)
        self.readerMcClf = rt.bookReaderMcClf(weightsMcClf,self.X)
        self.scl_center=scl_center
        self.scl_iqr=scl_iqr

    def shiftY(self):
        
        r=np.random.uniform()
        
        pPeak_data=(self.readerDataClf.EvaluateMVA("dataClf"))/2
        pPeak_mc=(self.readerMcClf.EvaluateMVA("mcClf")+1)/2

        drats=[]
        drats[0]=((1-pPeak_data)-(1-pPeak_mc))/pPeak_mc
        drats[1]=((pPeak_data)-(pPeak_mc))/(1-pPeak_mc)
        
        if self.X.qRC_Input_phoIso_ == 0. and (1-pPeak_data)>(1-pPeak_mc) and r<drats[0]:
            shifted = self.p2t()
        elif self.X.qRC_Input_phoIso_ > 0. and pPeak_data>pPeak_mc and r<drats[1]:
            shifted = 0.
        else:
            shifted = self.X.qRC_Input_phoIso_

        return shifted

    def p2t(self):
        
        return self.readerTailReg.EvaluateRegression("tailReg")
        
    def __call__(self,row):

        self.X.qRC_Input_pt_ = row[0]
        self.X.qRC_Input_ScEta_ = row[1]
        self.X.qRC_Input_Phi_ = row[2]
        self.X.qRC_Input_rho_ = row[3]
        self.X.qRC_Input_phoIso_ = row[4]
        self.X.qRC_Input_rand01_ = np.random.uniform(0.01,0.99)

        shifted = self.shiftY()
        if shifted == 0.:
            return shifted
        elif shifted > 0.:
            return shifted + self.scl_iqr*self.readerFinalReg.EvaluateRegression("finalReg")+self.scl_center



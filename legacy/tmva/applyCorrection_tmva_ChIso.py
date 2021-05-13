import pandas as pd
import numpy as np
import ROOT as rt

class computeCorrection_tmva_ChIso:

    def __init__(self,scl_centerChI,scl_iqrChI,scl_centerChIW,scl_iqrChIW,weightsFinalRegChI,weightsFinalRegChIW,weightsFinalTailRegChI,weightsFinalTailRegChIW,weightsDataClf,weightsMcClf,leg2016=False):
    
        rt.gROOT.LoadMacro("/mnt/t3nfs01/data01/shome/threiten/QReg/qRC/qRC_xmlReader_ChIso.C")

        self.X = rt.qRC_Input_ChIso()
        self.readerFinalRegChI = rt.bookReaderFinalReg(weightsFinalRegChI, self.X)
        self.readerTailRegChI = rt.bookReaderTailRegChIso("ChIso03",weightsFinalTailRegChI, self.X)
        self.readerFinalRegChIW = rt.bookReaderFinalReg(weightsFinalRegChIW, self.X)
        self.readerTailRegChIW = rt.bookReaderTailRegChIso("ChIso03worst",weightsFinalTailRegChIW, self.X)
        self.readerDataClf = rt.bookReader3CatClf(weightsDataClf, self.X)
        self.readerMcClf = rt.bookReader3CatClf(weightsMcClf,self.X)
        self.scl_centers=[scl_centerChI,scl_centerChIW]
        self.scl_iqrs=[scl_iqrChI,scl_iqrChIW]

    def shiftY(self):
        
        r=np.random.uniform()
        p=np.random.uniform()
        
        p_mc = self.readerMcClf.EvaluateMulticlass("3CatClf")
        p00_mc = p_mc[0]
        p01_mc = p_mc[1]
        p11_mc = p_mc[2]

        p_data = self.readerDataClf.EvaluateMulticlass("3CatClf")
        p00_data = p_data[0]
        p01_data = p_data[1]
        p11_data = p_data[2]

        if self.X.qRC_Input_chIso03_ == 0. and self.X.qRC_Input_chIso03worst_ == 0. and p00_mc > p00_data and r<=self.w(p00_mc,p00_data):
            if p01_mc<p01_data and p11_mc>p11_data:
                shifted = np.array([0.,self.p2t()[1]])
            elif p01_mc>p01_data and p11_mc<p11_data:
                shifted = self.p2t()
            elif p01_mc<p01_data and p11_mc<p11_data:
                if p<=self.z(p01_mc,p01_data,p00_mc,p00_data):
                    shifted = np.array([self.X.qRC_Input_chIso03_,self.p2t()[1]])
                else:
                    shifted = self.p2t()
                    
        elif self.X.qRC_Input_chIso03_ == 0. and self.X.qRC_Input_chIso03worst_ > 0. and p01_mc > p01_data and r<=self.w(p01_mc,p01_data):
            if p00_mc<p00_data and p11_mc>p11_data:
                shifted = np.zeros(2)
            elif p00_mc>p00_data and p11_mc<p11_data:
                shifted = np.array([self.p2t()[0],self.X.qRC_Input_chIso03worst_])
            elif p00_mc<p00_data and p11_mc<p11_data:
                if p<=self.z(p00_mc,p00_data,p01_mc,p01_data):
                    shifted = np.zeros(2)
                else:
                    shifted = np.array([self.p2t()[0],self.X.qRC_Input_chIso03worst_])
        
        elif self.X.qRC_Input_chIso03_ > 0. and self.X.qRC_Input_chIso03worst_ > 0. and p11_mc > p11_data and r<=self.w(p11_mc,p11_data):
            
            if p00_mc<p00_data and p01_mc>p01_data:
                shifted = np.zeros(2)
            elif p00_mc>p00_data and p01_mc<p01_data:    
                shifted = np.array([0.,self.X.qRC_Input_chIso03worst_])
            elif p00_mc<p00_data and p01_mc<p01_data:
                if p<=self.z(p00_mc,p00_data,p11_mc,p11_data):
                    shifted=np.zeros(2)
                else:
                    shifted=np.array([0.,self.X.qRC_Input_chIso03worst_])
        else:
            shifted = np.array([self.X.qRC_Input_chIso03_,self.X.qRC_Input_chIso03worst_])

        #print 'Output self.shifted: ', shifted
        return shifted

    def p2t(self):
        
        #print 'Peak to Tail called'
        return np.array([self.readerTailRegChI.EvaluateRegression("tailReg")[0],self.readerTailRegChIW.EvaluateRegression("tailReg")[0]])
 
    def w(self,p_mc,p_data):
        return 1.-np.divide(p_data,p_mc)
    
    def z(self,pj_mc,pj_data,pi_mc,pi_data):
        return np.divide(pj_data-pj_mc,pi_mc-pi_data)

    def __call__(self,row):
        
        self.X.qRC_Input_pt_ = row[0]
        self.X.qRC_Input_ScEta_ = row[1]
        self.X.qRC_Input_Phi_ = row[2]
        self.X.qRC_Input_rho_ = row[3]
        self.X.qRC_Input_chIso03_ = row[4]
        self.X.qRC_Input_chIso03worst_=row[5]
        # Conditional CDF from quantileRegression with first quantile 0.01 and last quantile 0.99
        self.X.qRC_Input_rand01_ = np.random.uniform(0.01,0.99)
        #print 'Input: ', self.X.qRC_Input_chIso03_, self.X.qRC_Input_chIso03worst_
        
        self.X.qRC_Input_chIso03_, self.X.qRC_Input_chIso03worst_ = self.shiftY()
        #print 'Shifted: ', self.X.qRC_Input_chIso03_, self.X.qRC_Input_chIso03worst_, '\n'

        if self.X.qRC_Input_chIso03_ == 0. and self.X.qRC_Input_chIso03worst_ == 0.:
            return self.X.qRC_Input_chIso03_, self.X.qRC_Input_chIso03worst_
        elif self.X.qRC_Input_chIso03_ == 0. and self.X.qRC_Input_chIso03worst_ > 0.:
            return self.X.qRC_Input_chIso03_, self.X.qRC_Input_chIso03worst_ + (self.readerFinalRegChIW.EvaluateRegression("finalReg")[0])*self.scl_iqrs[1] + self.scl_centers[1]
        elif self.X.qRC_Input_chIso03_ > 0. and self.X.qRC_Input_chIso03worst_ > 0.:
            return self.X.qRC_Input_chIso03_ + (self.readerFinalRegChI.EvaluateRegression("finalReg")[0])*self.scl_iqrs[0] + self.scl_centers[0], self.X.qRC_Input_chIso03worst_ + (self.readerFinalRegChIW.EvaluateRegression("finalReg")[0])*self.scl_iqrs[1] + self.scl_centers[1]

def applyCorrection_tmva_ChIso(df,scalerChI,scalerChIW,weightsFinalRegChI,weightsFinalRegChIW,weightsFinalTailRegChI,weightsFinalTailRegChIW,weightsDataClf,weightsMcClf,leg2016):

    columns = ["probePt","probeScEta","probePhi","rho","probeChIso03","probeChIso03worst"]
    row = df[columns].values
    correction = np.apply_along_axis(computeCorrection_tmva_ChIso(scalerChI.center_[0],scalerChI.scale_[0],scalerChIW.center_[0],scalerChIW.scale_[0],weightsFinalRegChI,weightsFinalRegChIW,weightsFinalTailRegChI,weightsFinalTailRegChIW,weightsDataClf,weightsMcClf,leg2016),1,row)
    #print correction
    df['probeChIso03_corr_tmva'] = correction[:,0]
    df['probeChIso03worst_corr_tmva'] = correction[:,1]

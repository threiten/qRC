import numpy as np

class Shifter:

    def __init__(self,mcp0tclf,datap0tclf,mcqclf,X,Y):
        
        proba_mc_clf = mcp0tclf.predict_proba(X)
        self.pPeak_mc = proba_mc_clf[:,0]
        self.pTail_mc = proba_mc_clf[:,1]
        
        proba_data_clf = datap0tclf.predict_proba(X)
        self.pPeak_data = proba_data_clf[:,0]
        self.pTail_data = proba_data_clf[:,1]
        
        self.mcqtls   = np.array([clf.predict(X) for clf in mcqclf])
        
        self.Y = Y

    def shiftYev(self,iev):
        
        Y = self.Y[iev]
        r=np.random.uniform()

        drats=self.get_diffrats(self.pPeak_mc[iev],self.pTail_mc[iev],self.pPeak_data[iev],self.pTail_data[iev])
        
        if Y == 0. and self.pTail_data[iev]>self.pTail_mc[iev] and r<drats[0]:
            Y_corr = self.p2t(iev)
        elif Y > 0. and self.pPeak_data[iev]>self.pPeak_mc[iev] and r<drats[1]:
            Y_corr = 0.
        else:
            Y_corr = Y

        return Y_corr

    def p2t(self,iev):
        
        epsilon = 1.e-5
        r=np.random.uniform(0.01+epsilon,0.99)
        bins = np.hstack(([0.01],np.linspace(0.05,0.95,19),[0.99]))

        indq = np.searchsorted(bins,r)
        y_tail = np.interp(r,bins[indq-1:indq+1],[self.mcqtls[indq-1,iev],self.mcqtls[indq,iev]])
        return y_tail

    def get_diffrats(self,pPeak_mc,pTail_mc,pPeak_data,pTail_data):
        return [np.divide(pTail_data-pTail_mc,pPeak_mc),np.divide(pPeak_data-pPeak_mc,pTail_mc)]

    def __call__(self):
        return np.array([self.shiftYev(iev) for iev in xrange(self.Y.size)]).ravel()


def applyShift(mcp0tclf,datap0tclf,tail_reg,X,Y):
    return Shifter(mcp0tclf,datap0tclf,tail_reg,X,Y)()

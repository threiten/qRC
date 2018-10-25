import numpy as np

class Shifter:

    def __init__(self,mcp0tclf,datap0tclf,tail_reg,X,Y):
        
        self.pPeak_mc = mcp0tclf.predict_proba(X)[0]
        self.pTail_mc = mcp0tclf.predict_proba(X)[1]

        self.pPeak_data = datap0tclf.predict_proba(X)[0]
        self.pTail_data = datap0tclf.predict_proba(X)[1]

        self.trail_reg = tail_reg.predict(X)
        
        self.Y = Y

    def shiftYev(self,iev):
        
        Y = self.Y[iev]
        r=np.random.uniform()

        drats=self.get_diffrats(self.pPeak_mc[iev],self.pTail_mc[iev],self.pPeak_data[iev],self.pTail_data[iev])

        if Y == 0 and pTail_data[iev]>pTail_mc[iev] and r>drats[0]:
            Y_corr = self.tail_reg[iev]
        elif Y > 0 and pPeak_data[iev]>pPeak_mc[iev] and r>drats[1]:
            Y_corr = 0
        else:
            Y_corr = Y

        return Y_corr

    def get_diffrats(pPeak_mc,pTail_mc,pPeak_data,pTail_data):
        return [((pTail_data-pTail_mc)/pPeak_mc),((pPeak_data-pPeak_mc)/pTail_mc)]

    def __call__(self):
        return np.array([self.shiftYev(iev) for iev in xrange(self.Y.size)]).ravel()


def applyShift(mcp0tclf,datap0tclf,tail_reg,X,Y):
    return Shifter(mcp0tclf,datap0tclf,tail_reg,X,Y)()

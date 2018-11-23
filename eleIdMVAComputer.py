import numpy as np
import ROOT as rt
from joblib import delayed, Parallel

class eleIdMvaComputer:

   def __init__(self,weightsEB1,weightsEB2,weightsEE,correct=[],tpC='qr',leg2016=False):
      rt.gROOT.LoadMacro("/mnt/t3nfs01/data01/shome/threiten/QReg/qRC/eleIDMVAonthefly.C")
            
      self.tpC = tpC
      self.leg2016 = leg2016
      self.X = rt.eleIDInput()
      self.readerEB1 = rt.bookReadersEB(weightsEB1, self.X, self.leg2016)
      self.readerEB2 = rt.bookReadersEB(weightsEB2, self.X, self.leg2016)
      self.readerEE = rt.bookReadersEE(weightsEE, self.X, self.leg2016)
      
     # print ("IdMvaComputer.__init__")
      if leg2016:
         columns = [ "probeSigmaIeIe", "probeCovarianceIphiIphi","probeFull5x5_e1x5","probeFull5x5_e5x5","probeR9", "probeEtaWidth", "probePhiWidth", "probeHoE", "ele_kfhits", "ele_kfchi2", "ele_gsfchi2", "ele_fbrem", "ele_gsfhits", "ele_expected_inner_hits", "ele_coversionVertexFitProbability", "ele_ep", "ele_eelepout", "ele_IoEmIop", "ele_deltaetain", "ele_deltaphiin", "ele_deltaetaseed", "probePhoIso", "probeChIso03", "probeNeutIso", "rho", "probeScEnergy","probeScPreshowerEnergy","probePt","probeScEta"]
      else:
         columns = [ "probeSigmaIeIe", "probeCovarianceIpIp","probeFull5x5_e1x5","probeFull5x5_e5x5","probeR9", "probeEtaWidth", "probePhiWidth", "probeHoE", "ele_kfhits", "ele_kfchi2", "ele_gsfchi2", "ele_fbrem", "ele_gsfhits", "ele_expected_inner_hits", "ele_coversionVertexFitProbability", "ele_ep", "ele_eelepout", "ele_IoEmIop", "ele_deltaetain", "ele_deltaphiin", "ele_deltaetaseed", "probePhoIso", "probeChIso03", "probeNeutIso", "rho", "probeScEnergy","probeScPreshowerEnergy","probePt","probeScEta"]         
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
 
      return np.apply_along_axis( self.predict, 1, Xvals ).ravel()
      
   def predict(self,row):
      return self.predictEB1(row) if np.abs(row[-1]) < 0.8 else (self.predictEB2(row) if np.abs(row[-1]) < 1.5 else self.predictEE(row)) 
      
   def predictEB1(self,row):
      
      self.X.ele_oldsigmaietaieta = row[0]
      self.X.ele_oldsigmaiphiiphi = row[1]
      e1x5 = row[2]
      e5x5 = row[3]
      self.X.ele_oldcircularity = 1-(e1x5/e5x5)
      self.X.ele_oldr9 = row[4]
      self.X.ele_scletawidth = row[5]
      self.X.ele_sclphiwidth = row[6]
      self.X.ele_oldhe = row[7]
      self.X.ele_kfhits = row[8]
      self.X.ele_kfchi2 = row[9]
      self.X.ele_gsfchi2 = row[10]
      self.X.ele_fbrem = row[11]
      self.X.ele_gsfhits = row[12]
      self.X.ele_expected_inner_hits = row[13]
      self.X.ele_conversionVertexFitProbability = row[14]
      self.X.ele_ep = row[15]
      self.X.ele_eelepout = row[16]
      self.X.ele_IoEmIop = row[17]
      self.X.ele_deltaetain = row[18]
      self.X.ele_deltaphiin = row[19]
      self.X.ele_deltaetaseed = row[20]
      self.X.ele_pfPhotonIso = row[21]
      self.X.ele_pfChargedHadIso = row[22]
      self.X.ele_pfNeutralHadIso = row[23]
      self.X.rho = row[24]
      self.X.ele_pt = row[27]
      self.X.scl_eta = row[28]

      return self.readerEB1.EvaluateMVA("BDT")

   def predictEB2(self,row):
      
      self.X.ele_oldsigmaietaieta = row[0]
      self.X.ele_oldsigmaiphiiphi = row[1]
      e1x5 = row[2]
      e5x5 = row[3]
      self.X.ele_oldcircularity = 1-(e1x5/e5x5)
      self.X.ele_oldr9 = row[4]
      self.X.ele_scletawidth = row[5]
      self.X.ele_sclphiwidth = row[6]
      self.X.ele_oldhe = row[7]
      self.X.ele_kfhits = row[8]
      self.X.ele_kfchi2 = row[9]
      self.X.ele_gsfchi2 = row[10]
      self.X.ele_fbrem = row[11]
      self.X.ele_gsfhits = row[12]
      self.X.ele_expected_inner_hits = row[13]
      self.X.ele_conversionVertexFitProbability = row[14]
      self.X.ele_ep = row[15]
      self.X.ele_eelepout = row[16]
      self.X.ele_IoEmIop = row[17]
      self.X.ele_deltaetain = row[18]
      self.X.ele_deltaphiin = row[19]
      self.X.ele_deltaetaseed = row[20]
      self.X.ele_pfPhotonIso = row[21]
      self.X.ele_pfChargedHadIso = row[22]
      self.X.ele_pfNeutralHadIso = row[23]
      self.X.rho = row[24]
      self.X.ele_pt = row[27]
      self.X.scl_eta = row[28]
       
      return self.readerEB2.EvaluateMVA("BDT")


   def predictEE(self,row):
      self.X.ele_oldsigmaietaieta = row[0]
      self.X.ele_oldsigmaiphiiphi = row[1]
      e1x5 = row[2]
      e5x5 = row[3]
      self.X.ele_oldcircularity = 1-(e1x5/e5x5)
      self.X.ele_oldr9 = row[4]
      self.X.ele_scletawidth = row[5]
      self.X.ele_sclphiwidth = row[6]
      self.X.ele_oldhe = row[7]
      self.X.ele_kfhits = row[8]
      self.X.ele_kfchi2 = row[9]
      self.X.ele_gsfchi2 = row[10]
      self.X.ele_fbrem = row[11]
      self.X.ele_gsfhits = row[12]
      self.X.ele_expected_inner_hits = row[13]
      self.X.ele_conversionVertexFitProbability = row[14]
      self.X.ele_ep = row[15]
      self.X.ele_eelepout = row[16]
      self.X.ele_IoEmIop = row[17]
      self.X.ele_deltaetain = row[18]
      self.X.ele_deltaphiin = row[19]
      self.X.ele_deltaetaseed = row[20]
      self.X.ele_pfPhotonIso = row[21]
      self.X.ele_pfChargedHadIso = row[22]
      self.X.ele_pfNeutralHadIso = row[23]
      self.X.rho = row[24]
      #print "IdMvaComputer.predictEE"
      ScEn = row[25]
      esEn = row[26]
      self.X.ele_psEoverEraw = esEn/ScEn
      self.X.ele_pt = row[27]
      self.X.scl_eta = row[28]

      return self.readerEE.EvaluateMVA("BDT")


def helpComputeEleIdMva(weightsEB1,weightsEB2,weightsEE,correct,X,tpC,leg2016):
   return eleIdMvaComputer(weightsEB1,weightsEB2,weightsEE,correct,tpC,leg2016)(X)

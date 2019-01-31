import quantileRegression_chain_disc as qRCd
import numpy as np
import argparse
import yaml

def main(options):

    stream = file(options.config,'r')
    inp=yaml.load(stream)

    dataframes = inp['dataframes']
    showerShapes = inp['showerShapes']
    chIsos = inp['chIsos']
    year = str(inp['year'])
    workDir = inp['workDir']
    weightsDirs = inp['weightsDirs']

    if year == '2017':
        cols=["mass","probeScEnergy","probeScEta","probePhi","run","weight","weight_clf","rho","probeR9","probeSigmaIeIe","probePhiWidth","probeEtaWidth","probeCovarianceIeIp","probeCovarianceIpIp","probeS4","probePhoIso","probeChIso03","probeChIso03worst","probeSigmaRR","probeScPreshowerEnergy","probePt"]
    elif year == '2016':
        cols=["mass","probeScEnergy","probeScEta","probePhi","run","weight","weight_clf","rho","probeR9","probeSigmaIeIe","probePhiWidth","probeEtaWidth","probeCovarianceIetaIphi","probeCovarianceIphiIphi","probeS4","probePhoIso","probeChIso03","probeChIso03worst","probeSigmaRR","probeScPreshowerEnergy","probePt"]


    qRC = qRCd.quantileRegression_chain(year,options.EBEE,workDir,showerShapes)
    qRC.loadMCDF(dataframes['mc'][options.EBEE]['input'],0,options.n_evts,columns=cols)
    qRC.loadDataDF(dataframes['data'][options.EBEE]['input'],0,options.n_evts,columns=cols)
    
    qRC.setupJoblib('all_6gb')
    for var in qRC.vars:
        qRC.loadClfs(var,weightsDir=weightsDirs['showerShapes'])
        qRC.correctY(var,n_jobs=20)

    qRC_PI = qRCd.quantileRegression_chain_disc(year,options.EBEE,workDir,['probePhoIso'])

    qRC_PI.MC = qRC.MC
    qRC_PI.data = qRC.data

    qRC_PI.loadp0tclf('probePhoIso',weightsDir=weightsDirs['phoIso'])
    qRC_PI.loadClfs('probePhoIso',weightsDir=weightsDirs['phoIso'])
    qRC_PI.setupJoblib('all_6gb')
    qRC_PI.correctY('probePhoIso',n_jobs=20)

    qRC_ChI = qRCd.quantileRegression_chain_disc(year,options.EBEE,workDir,chIsos)

    qRC_ChI.MC = qRC_PI.MC
    qRC_ChI.data = qRC_PI.data

    qRC_ChI.load3Catclf(qRC_ChI.vars,weightsDir=weightsDirs['chIsos'])
    qRC_ChI.loadTailRegressors(qRC_ChI.vars,weightsDir=weightsDirs['chIsos'])
    qRC_ChI.setupJoblib('all_6gb')
    for var in qRC_ChI.vars:
        qRC_ChI.loadClfs(var,weightsDir=weightsDirs['chIsos'])
        qRC_ChI.correctY(var,n_jobs=20)

    qRC_ChI.MC.to_hdf('{}/{}'.format(workDir,dataframes['mc'][options.EBEE]['output']),'df',mode='w',format='t')
  
if __name__=="__main__":
     parser=argparse.ArgumentParser()
     requiredArgs = parser.add_argument_group('Required Arguments')
     requiredArgs.add_argument('-c','--config', action='store', default='quantile_config.yaml', type=str,required=True)
     requiredArgs.add_argument('-E','--EBEE', action='store', type=str, required=True)
     requiredArgs.add_argument('-N','--n_evts', action='store', type=int, required=True)
     
     options=parser.parse_args()
     main(options)

from quantile_regression_chain import quantileRegression_chain_disc
from quantile_regression_chain import quantileRegression_chain
import numpy as np
import argparse
import yaml
import root_pandas

import logging
logger = logging.getLogger("")

def setup_logging(level=logging.DEBUG):
    logger.setLevel(level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def main(options):

    stream = open(options.config,'r')
    inp=yaml.safe_load(stream)

    dataframes = inp['dataframes']
    showerShapes = inp['showerShapes']
    chIsos = inp['chIsos']
    year = str(inp['year'])
    workDir = inp['workDir']
    weightsDirs = inp['weightsDirs']
    finalWeightsDirs = inp['finalWeightsDirs']

    if year == '2017':
        cols=["mass","probeScEnergy","probeScEta","probePhi","run","weight","weight_clf","rho","probeR9","probeSigmaIeIe","probePhiWidth","probeEtaWidth","probeCovarianceIeIp","probeCovarianceIpIp","probeS4","probePhoIso","probeChIso03","probeChIso03worst","probeSigmaRR","probeScPreshowerEnergy","probePt","tagPt","probePassEleVeto","tagScEta","probePass_invEleVeto"]
        treenameMC = '/tagAndProbeDumper/trees/DYJetsToLL_amcatnloFXFX_13TeV_All'
    elif year == '2018':
        cols=["mass","probeScEnergy","probeScEta","probePhi","run","weight","weight_clf","rho","probeR9","probeSigmaIeIe","probePhiWidth","probeEtaWidth","probeCovarianceIeIp","probeCovarianceIpIp","probeS4","probePhoIso","probeChIso03","probeChIso03worst","probeSigmaRR","probePt","tagPt","probePassEleVeto","tagScEta"]
        treenameMC = '/tagAndProbeDumper/trees/DYJetsToLL_amcatnloFXFX_13TeV_All'
    elif year == '2016':
        cols=["mass","probeScEnergy","probeScEta","probePhi","run","weight","weight_clf","rho","probeR9","probeSigmaIeIe","probePhiWidth","probeEtaWidth","probeCovarianceIetaIphi","probeCovarianceIphiIphi","probeS4","probePhoIso","probeChIso03","probeChIso03worst","probeSigmaRR","probeScPreshowerEnergy","probePt","tagPt","probePassEleVeto","tagScEta","probePass_invEleVeto"]
        treenameMC = '/tagAndProbeDumper/trees/DYJets_madgraph_13TeV_All'

    EBEE_cut = 'abs(probeScEta)<1.4442' if options.EBEE == 'EB' else 'abs(probeScEta)>1.556'

    qRC = quantileRegression_chain(year,options.EBEE,workDir,showerShapes)
    if dataframes['mc'][options.EBEE]['input'].split('.')[-1] == 'root':
        cols.pop(cols.index("weight_clf"))
        qRC.MC = root_pandas.read_root(dataframes['mc'][options.EBEE]['input'],treenameMC,columns=cols).query(EBEE_cut)
    else:
        qRC.loadMCDF(dataframes['mc'][options.EBEE]['input'],0,options.n_evts,columns=cols)

    if dataframes['data'][options.EBEE]['input'].split('.')[-1] == 'root':
        if "weight_clf" in cols:
            cols.pop(cols.index("weight_clf"))
        qRC.data = root_pandas.read_root(dataframes['data'][options.EBEE]['input'],'/tagAndProbeDumper/trees/Data_13TeV_All',columns=cols).query(EBEE_cut)
    else:
        qRC.loadDataDF(dataframes['data'][options.EBEE]['input'],0,options.n_evts,columns=cols)

    if options.backend is not None:
        qRC.setupJoblib_ray(cluster_id = options.clusterid)
    for var in qRC.vars:
        if options.final:
            qRC.loadFinalRegression(var,weightsDir=finalWeightsDirs['showerShapes'])
            qRC.loadScaler(var,weightsDir=finalWeightsDirs['showerShapes'])
            qRC.applyFinalRegression(var)
        qRC.loadClfs(var,weightsDir=weightsDirs['showerShapes'])
        print('Correcting variables')
        qRC.correctY(var,n_jobs=options.n_jobs)

    qRC_PI = quantileRegression_chain_disc(year,options.EBEE,workDir,['probePhoIso'])

    qRC_PI.MC = qRC.MC
    qRC_PI.data = qRC.data

    qRC_PI.loadp0tclf('probePhoIso',weightsDir=weightsDirs['phoIso'])
    if options.backend is not None:
        qRC_PI.setupJoblib_ray(cluster_id = options.clusterid)
    if options.final:
        qRC_PI.loadFinalRegression('probePhoIso',weightsDir=finalWeightsDirs['phoIso'])
        qRC_PI.loadFinalTailRegressor('probePhoIso',weightsDir=finalWeightsDirs['phoIso'])
        qRC_PI.loadScaler('probePhoIso',weightsDir=finalWeightsDirs['phoIso'])
        qRC_PI.applyFinalRegression('probePhoIso',n_jobs=options.n_jobs)
    qRC_PI.loadClfs('probePhoIso',weightsDir=weightsDirs['phoIso'])
    qRC_PI.correctY('probePhoIso',n_jobs=options.n_jobs)

    qRC_ChI = quantileRegression_chain_disc(year,options.EBEE,workDir,chIsos)

    qRC_ChI.MC = qRC_PI.MC
    qRC_ChI.data = qRC_PI.data

    qRC_ChI.load3Catclf(qRC_ChI.vars,weightsDir=weightsDirs['chIsos'])
    if options.backend is not None:
        qRC_ChI.setupJoblib_ray(cluster_id = options.clusterid)
    if options.final:
        qRC_ChI.loadFinalTailRegressor(qRC_ChI.vars,weightsDir=finalWeightsDirs['chIsos'])
        for var in qRC_ChI.vars:
            qRC_ChI.loadFinalRegression(var,weightsDir=finalWeightsDirs['chIsos'])
            qRC_ChI.loadScaler(var,weightsDir=finalWeightsDirs['chIsos'])
            qRC_ChI.applyFinalRegression(var,n_jobs=options.n_jobs)
    qRC_ChI.loadTailRegressors(qRC_ChI.vars,weightsDir=weightsDirs['chIsos'])
    for var in qRC_ChI.vars:
        qRC_ChI.loadClfs(var,weightsDir=weightsDirs['chIsos'])
        qRC_ChI.correctY(var,n_jobs=options.n_jobs)

    if options.mvas:
        if year == '2017':
            weights = ("/mnt/t3nfs01/data01/shome/threiten/QReg/ReReco17_data/camp_3_1_0/PhoIdMVAweights/HggPhoId_94X_barrel_BDT_v2.weights.xml","/mnt/t3nfs01/data01/shome/threiten/QReg/ReReco17_data/camp_3_1_0/PhoIdMVAweights/HggPhoId_94X_endcap_BDT_v2.weights.xml")
            leg2016=False
        if year == '2018':
            leg2016=False
            weights = ('/work/gallim/weights/id_mva/HggPhoId_94X_barrel_BDT_v2.weights.xml', '/work/gallim/weights/id_mva/HggPhoId_94X_endcap_BDT_v2.weights.xml')
            if options.EBEE == 'EB':
                qRC_ChI.data['probeScPreshowerEnergy'] = -999.*np.ones(qRC_ChI.data.index.size)
                qRC_ChI.MC['probeScPreshowerEnergy'] = -999.*np.ones(qRC_ChI.MC.index.size)
            elif options.EBEE == 'EE':
                qRC_ChI.data['probeScPreshowerEnergy'] = np.zeros(qRC_ChI.data.index.size)
                qRC_ChI.MC['probeScPreshowerEnergy'] = np.zeros(qRC_ChI.MC.index.size)
        elif year == '2016':
            weights = ("/mnt/t3nfs01/data01/shome/threiten/QReg/ReReco16/PhoIdMVAweights/HggPhoId_barrel_Moriond2017_wRhoRew.weights.xml","/mnt/t3nfs01/data01/shome/threiten/QReg/ReReco16/PhoIdMVAweights/HggPhoId_endcap_Moriond2017_wRhoRew.weights.xml")
            leg2016=True
        if options.final:
            mvas = [ ("newPhoID","data",[]), ("newPhoIDcorrAll","qr",qRC.vars + qRC_PI.vars + qRC_ChI.vars), ("newPhoIDcorrAllFinal","final",qRC.vars + qRC_PI.vars + qRC_ChI.vars)]
        else:
            mvas = [ ("newPhoID","data",[]), ("newPhoIDcorrAll","qr",qRC.vars + qRC_PI.vars + qRC_ChI.vars)]

        qRC.computeIdMvas( mvas[:1],  weights,'data', n_jobs=options.n_jobs, leg2016=leg2016)
        qRC.computeIdMvas( mvas, weights,'mc', n_jobs=options.n_jobs , leg2016=leg2016)

    qRC_ChI.MC.to_hdf('{}/{}'.format(workDir,dataframes['mc'][options.EBEE]['output']),'df',mode='w',format='t')
    if options.mvas:
        qRC_ChI.data.to_hdf('{}/{}'.format(workDir,dataframes['data'][options.EBEE]['output']),'df',mode='w',format='t')

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguments')
    requiredArgs.add_argument('-c','--config', action='store', default='quantile_config.yaml', type=str,required=True)
    requiredArgs.add_argument('-E','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-N','--n_evts', action='store', type=int, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-B','--backend', action='store', type=str)
    optArgs.add_argument('-i','--clusterid', action='store', type=str)
    optArgs.add_argument('-f','--final', action='store_true', default=False)
    optArgs.add_argument('-m','--mvas', action='store_true', default=False)
    optArgs.add_argument('-n','--n_jobs', action='store', type=int, default=1)
    options=parser.parse_args()
    setup_logging(logging.INFO)
    main(options)

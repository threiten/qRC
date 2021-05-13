from quantile_regression_chain import quantileRegression_chain_disc
from quantile_regression_chain import quantileRegression_chain
import numpy as np
import pandas as pd
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

    preshower_variable = 'phoIdMVA_esEnovSCRawEn'
    cols = ['probeScEta', 'probeEtaWidth', 'probeR9', 'weight', 'probeSigmaRR', 'tagChIso03', 'probeChIso03', 'probeS4', 'tagR9', 'tagPhiWidth_Sc', 'probePt', 'tagSigmaRR', 'probePhiWidth', 'probeChIso03worst', 'puweight', 'tagEleMatch', 'tagPhi', 'probeScEnergy', 'nvtx', 'probePhoIso', 'probePhoIso03', 'tagPhoIso', 'run', 'tagScEta', 'probeEleMatch', 'probeCovarianceIeIp', 'tagPt', 'rho', 'tagS4', 'tagSigmaIeIe', 'tagCovarianceIpIp', 'tagCovarianceIeIp', 'tagScEnergy', 'tagChIso03worst', 'probeSigmaIeIe', 'probePhi', 'mass', 'probeCovarianceIpIp', 'tagEtaWidth_Sc', 'probeHoE', 'probeFull5x5_e1x5', 'probeFull5x5_e5x5', 'probeNeutIso', 'probePassEleVeto', 'phoIdMVA_esEnovSCRawEn']

    ss = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']

    qRC = quantileRegression_chain(year,options.EBEE,workDir,[preshower_variable])
    qRC.kinrho += ss
    qRC.loadMCDF(dataframes['mc'][options.EBEE]['input'],0,options.n_evts,columns=cols)
    qRC.loadDataDF(dataframes['data'][options.EBEE]['input'],0,options.n_evts,columns=cols)

    if options.backend is not None:
        qRC.setupJoblib_ray(cluster_id = options.clusterid)
    if options.final:
        qRC.loadFinalRegression(preshower_variable,weightsDir=finalWeightsDirs)
        qRC.loadScaler(preshower_variable,weightsDir=finalWeightsDirs)
        #qRC.applyFinalRegression(preshower_variable,n_jobs=options.n_jobs)
        qRC.applyFinalRegression(preshower_variable)
    qRC.loadClfs(preshower_variable,weightsDir=weightsDirs)
    qRC.correctY(preshower_variable,n_jobs=options.n_jobs)


    qRC.MC.to_hdf('{}/{}'.format(workDir,dataframes['mc'][options.EBEE]['output']),'df',mode='w',format='t')

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
    optArgs.add_argument('-n','--n_jobs', action='store', type=int, default=1)
    options=parser.parse_args()
    setup_logging(logging.INFO)
    main(options)

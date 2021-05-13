from quantile_regression_chain import quantileRegression_chain_disc
from quantile_regression_chain import quantileRegression_chain
import numpy as np
import pandas as pd
import argparse
import yaml

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

    # Flashgg PhoIso is training PhoIso03, hence use PhoIso since we are dealing with the quantity dumped directly by flashgg
    # All the previous variables (i.e. SS + ChIso + PhIso) to which corrections have to be applied have '_uncorr' as a suffix
    # Since we have to recompute thw corrected quantities with the specific regressors trained for each systematic, we import only the uncorrected ones

    # NB: remember to make sure that the preshower variable name is correct (open the h5 data and mc files and check)

    additional_cols = ["mass","probeScEnergy","probeScEta","probePhi","run","weight","rho","probeSigmaRR","probePt","tagPt","probePassEleVeto","tagScEta"]

    pho_iso = ['probePhoIso']
    preshower = ['probeesEnergyOverSCRawEnergy']

    previous_vars = showerShapes + chIsos + pho_iso
    uncorr_previous_vars = [var + '_uncorr' for var in previous_vars]

    logger.info("previous vars are {}".format(previous_vars))
    logger.info("uncorrected previous vars are {}".format(uncorr_previous_vars))

    EBEE_cut = 'abs(probeScEta)<1.4442' if options.EBEE == 'EB' else 'abs(probeScEta)>1.556'

    ## SS
    logger.info("Starting SS")
    qRC = quantileRegression_chain(year,options.EBEE,workDir,showerShapes)
    qRC.loadMCDF(dataframes['mc'][options.EBEE]['input'],0,options.n_evts,columns=additional_cols + uncorr_previous_vars + preshower)
    qRC.loadDataDF(dataframes['data'][options.EBEE]['input'],0,options.n_evts,columns=additional_cols + uncorr_previous_vars + preshower)

    qRC.MC.rename(columns=dict(zip(uncorr_previous_vars, previous_vars)), inplace=True)
    qRC.data.rename(columns=dict(zip(uncorr_previous_vars, previous_vars)), inplace=True)

    if options.backend is not None:
        qRC.setupJoblib_ray(cluster_id = options.clusterid)
    for var in qRC.vars:
        qRC.loadClfs(var,weightsDir=weightsDirs['prev'])
        qRC.correctY(var,n_jobs=options.n_jobs)

    # PhoIso
    logger.info("Starting PhoIso")
    qRC_PI = quantileRegression_chain_disc(year,options.EBEE,workDir, pho_iso)

    qRC_PI.MC = qRC.MC
    qRC_PI.data = qRC.data

    qRC_PI.loadp0tclf('probePhoIso',weightsDir=weightsDirs['prev'])
    if options.backend is not None:
        qRC_PI.setupJoblib_ray(cluster_id = options.clusterid)
    qRC_PI.loadClfs('probePhoIso',weightsDir=weightsDirs['prev'])
    qRC_PI.correctY('probePhoIso',n_jobs=options.n_jobs)

    # ChIso
    logger.info("Starting ChIso")
    qRC_ChI = quantileRegression_chain_disc(year,options.EBEE,workDir,chIsos)

    qRC_ChI.MC = qRC_PI.MC
    qRC_ChI.data = qRC_PI.data

    qRC_ChI.load3Catclf(qRC_ChI.vars,weightsDir=weightsDirs['prev'])
    if options.backend is not None:
        qRC_ChI.setupJoblib_ray(cluster_id = options.clusterid)
    qRC_ChI.loadTailRegressors(qRC_ChI.vars,weightsDir=weightsDirs['prev'])
    for var in qRC_ChI.vars:
        qRC_ChI.loadClfs(var,weightsDir=weightsDirs['prev'])
        qRC_ChI.correctY(var,n_jobs=options.n_jobs)

    # Preshower
    logger.info("Starting preshower")
    qRC_pres = quantileRegression_chain(year,options.EBEE,workDir,preshower)
    qRC_pres.kinrho += [var + '_corr' for var in showerShapes]
    qRC_pres.MC = qRC_ChI.MC
    qRC_pres.data = qRC_ChI.data

    if options.backend is not None:
        qRC_pres.setupJoblib_ray(cluster_id = options.clusterid)
    qRC_pres.loadClfs(preshower[0],weightsDir=weightsDirs['preshower'])
    qRC_pres.correctY(preshower[0],n_jobs=options.n_jobs)


    if options.mvas:
        # Now the corrected variables have the suffix _corr, the uncorrected ones have nothing
        logger.info("Start recomputing PhoID")
        leg2016=False
        weights = ('/work/gallim/weights/id_mva/HggPhoId_94X_barrel_BDT_v2.weights.xml', '/work/gallim/weights/id_mva/HggPhoId_94X_endcap_BDT_v2.weights.xml')

        # Compute PhoID both for uncorrected and corrected vars in MC
        mvas = [ ("newPhoID","data",[]), ("newPhoIDcorrAll", "qr", qRC.vars + qRC_PI.vars + qRC_ChI.vars + qRC_pres.vars)]

        qRC_pres.computeIdMvas(mvas, weights,'mc', n_jobs=options.n_jobs , leg2016=leg2016)

    qRC_pres.MC.to_hdf('{}/{}'.format(workDir,dataframes['mc'][options.EBEE]['output']),'df',mode='w',format='t')

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

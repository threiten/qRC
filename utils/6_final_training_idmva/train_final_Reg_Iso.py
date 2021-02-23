import argparse
import yaml
from quantile_regression_chain import quantileRegression_chain_disc as QReg_D

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

    df_name_data = inp['dataframes']['data_{}'.format(options.EBEE)]
    df_name_mc = inp['dataframes']['mc_{}'.format(options.EBEE)]
    variables = inp['variables']
    year = str(inp['year'])
    workDir = inp['workDir']
    weightsDir = inp['weightsDir']
    outDir = inp['outDir']

    if options.split is not None:
        print('Using split dfs for training two sets of weights!')
        df_name_data = df_name_data + '_spl{}_1M.h5'.format(options.split)
        df_name_mc = df_name_mc + '_spl{}_1M.h5'.format(options.split)
        weightsDir = weightsDir + '/spl{}'.format(options.split)
        outDir = outDir + '/spl{}'.format(options.split)

    columns = ['probePt','probeScEta','probePhi','rho'] + variables

    qRC = QReg_D(year,options.EBEE,workDir,variables)
    qRC.loadMCDF(df_name_mc,0,options.n_evts,rsh=False,columns=columns)

    if options.backend is not None:
        qRC.setupJoblib(cluster_id=options.clusterid)

    for var in qRC.vars:
        qRC.trainFinalTailRegressor(var, weightsDir=outDir, weightsDirIn=weightsDir, n_jobs=options.n_jobs)
        if len(variables) == 1:
            qRC.loadp0tclf(variables[0],weightsDir=weightsDir)
        elif len(variables) > 1:
            qRC.load3Catclf(variables, weightsDir = weightsDir)
        qRC.loadClfs(var, weightsDir = weightsDir)
        qRC.correctY(var, n_jobs = options.n_jobs)
        qRC.trainFinalRegression(var, weightsDir=outDir, n_jobs=options.n_jobs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-E','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-N','--n_evts', action='store', type=int, required=True)
    requiredArgs.add_argument('-c','--config', action='store', type=str, required=True)
    optionalArgs = parser.add_argument_group('Optional Arguements')
    optionalArgs.add_argument('-n','--n_jobs', action='store', default=1, type=int)
    optionalArgs.add_argument('-s','--split', action='store', type=int)
    optionalArgs.add_argument('-B','--backend', action='store', type=str)
    optionalArgs.add_argument('-i','--clusterid', action='store', type=str)
    options=parser.parse_args()
    setup_logging(logging.INFO)
    main(options)

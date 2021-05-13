import argparse
import yaml
from quantile_regression_chain import quantileRegression_chain as QReg_C

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

    if options.split is not None:
        print('Using split dfs for training two sets of weights!')
        df_name_data = df_name_data.split('.')[0] + '_spl{}.h5'.format(options.split)
        df_name_mc = df_name_mc.split('.')[0] + '_spl{}.h5'.format(options.split)
        weightsDir = weightsDir + '/spl{}'.format(options.split)

    ss = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    qRC = QReg_C(year,options.EBEE,workDir,variables)
    qRC.kinrho += ss
    if options.backend is not None:
        qRC.setupJoblib_ray(cluster_id=options.clusterid)
    qRC.loadMCDF(df_name_mc,0,options.n_evts,rsh=False)
    # qRC.loadDataDF(df_name_data,0,options.n_evts,rsh=False)
    print(qRC.MC.index.size)
    qRC.trainAllMC(weightsDir=weightsDir,n_jobs=21)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-E','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-c','--config', action='store', type=str, required=True)
    requiredArgs.add_argument('-N','--n_evts', action='store', type=int, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-s','--split', action='store', type=int)
    optArgs.add_argument('-B','--backend', action='store', type=str)
    optArgs.add_argument('-i','--clusterid', action='store', type=str)
    options=parser.parse_args()
    setup_logging(logging.INFO)
    main(options)

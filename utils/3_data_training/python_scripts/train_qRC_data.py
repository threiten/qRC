import argparse
import yaml
import os
from quantile_regression_chain import quantileRegression_chain as QReg_C



def main(options):

    stream = open(options.config,'r')
    inp=yaml.safe_load(stream)

    df_name = inp['dataframes']['data_{}'.format(options.EBEE)]
    variables = inp['variables']
    year = str(inp['year'])
    workDir = inp['workDir']
    weightsDir = inp['weightsDir']

    if options.split is not None:
        print ('Using split dfs for training two sets of weights!')
        df_name = df_name.replace('.h5', '') + '_spl{}.h5'.format(options.split)
        weightsDir = weightsDir + '/spl{}'.format(options.split)

    if year == '2017' or year == '2018':
        columns = ['probePt','probeScEta','probePhi','rho','probeEtaWidth','probeSigmaIeIe','probePhiWidth','probeR9','probeS4','probeCovarianceIeIp']
    elif year == '2016':
        columns = ['probePt','probeScEta','probePhi','rho','probeEtaWidth','probeSigmaIeIe','probePhiWidth','probeR9','probeS4','probeCovarianceIetaIphi']
    qRC = QReg_C(year,options.EBEE,workDir,variables)
    qRC.quantiles = [options.quantile]
    qRC.loadDataDF(df_name,0,options.n_evts,rsh=False,columns=columns)
    if not os.path.exists(weightsDir + 'data_weights_{}_{}_{}.pkl'.format(
        options.EBEE, options.variable, str(options.quantile).replace('.', 'p'))):
        qRC.trainOnData(options.variable,weightsDir=weightsDir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-q','--quantile', action='store', type=float, required=True)
    requiredArgs.add_argument('-v','--variable', action='store', type=str, required=True)
    requiredArgs.add_argument('-E','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-N','--n_evts', action='store', type=int, required=True)
    requiredArgs.add_argument('-c','--config', action='store', type=str, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-s','--split', action='store', type=int)
    options=parser.parse_args()
    main(options)


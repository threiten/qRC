from import_file import import_file
import argparse
import yaml

QReg_C = import_file("/mnt/t3nfs01/data01/shome/threiten/QReg/qRC/quantileRegression_chain")


def main(options):

    stream = file(options.config,'r')
    inp=yaml.load(stream)

    df_name = inp['dataframes']['data_{}'.format(options.EBEE)]
    variables = inp['variables']
    year = str(inp['year'])
    workDir = inp['workDir']
    weightsDir = inp['weightsDir']

    if year == '2017':
        columns = ['probePt','probeScEta','probePhi','rho','probeEtaWidth','probeSigmaIeIe','probePhiWidth','probeR9','probeS4','probeCovarianceIeIp']
    elif year == '2016':
        columns = ['probePt','probeScEta','probePhi','rho','probeEtaWidth','probeSigmaIeIe','probePhiWidth','probeR9','probeS4','probeCovarianceIetaIphi']
    qRC = QReg_C.quantileRegression_chain(year,options.EBEE,workDir,variables)
    qRC.quantiles = [options.quantile]
    qRC.loadDataDF(df_name,0,options.n_evts,rsh=False,columns=columns)
    qRC.trainOnData(options.variable,weightsDir=weightsDir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-q','--quantile', action='store', type=float, required=True)
    requiredArgs.add_argument('-v','--variable', action='store', type=str, required=True)
    requiredArgs.add_argument('-E','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-N','--n_evts', action='store', type=int, required=True)
    requiredArgs.add_argument('-c','--config', action='store', type=str, required=True)
    options=parser.parse_args()
    main(options)

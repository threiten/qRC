from import_file import import_file
import argparse
import yaml

QReg_C = import_file("/mnt/t3nfs01/data01/shome/threiten/QReg/qRC/quantileRegression_chain")

def main(options):

    stream = file(options.config,'r')
    inp=yaml.load(stream)

    variables = inp['variables']
    year = str(inp['year'])
    workDir = inp['workDir']
    weightsDir = inp['weightsDir']
    outDir = inp['outDir']

    if year == '2017':
        columns = ['probePt','probeScEta','probePhi','rho','probeEtaWidth','probeSigmaIeIe','probePhiWidth','probeR9','probeS4','probeCovarianceIeIp']#,'probeEtaWidth_corr','probeSigmaIeIe_corr','probePhiWidth_corr','probeR9_corr','probeS4_corr','probeCovarianceIeIp_corr']
    elif year == '2016':
        columns = ['probePt','probeScEta','probePhi','rho','probeEtaWidth','probeSigmaIeIe','probePhiWidth','probeR9','probeS4','probeCovarianceIetaIphi']#,'probeEtaWidth_corr','probeSigmaIeIe_corr','probePhiWidth_corr','probeR9_corr','probeS4_corr','probeCovarianceIeIp_corr']

    qRC = QReg_C.quantileRegression_chain(year,options.EBEE,workDir,variables)
    qRC.loadMCDF(inp['dataframes']['mc_{}'.format(options.EBEE)],0,options.n_evts,rsh=False,columns=columns)
    qRC.loadDataDF(inp['dataframes']['data_{}'.format(options.EBEE)],0,options.n_evts,rsh=False,columns=columns)
    for var in qRC.vars:
        qRC.loadClfs(var,weightsDir)
        qRC.correctY(var,n_jobs=options.n_jobs)
        qRC.trainFinalRegression(var,weightsDir=outDir,n_jobs=options.n_jobs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-v','--variable', action='store', type=str, required=True)
    requiredArgs.add_argument('-E','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-N','--n_evts', action='store', type=int, required=True)
    requiredArgs.add_argument('-c','--config', action='store', type=str, required=True)
    optionalArgs = parser.add_argument_group('Optional Arguements')
    optionalArgs.add_argument('-n','--n_jobs', action='store', type=int)
    options=parser.parse_args()
    main(options)




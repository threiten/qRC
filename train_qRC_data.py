from import_file import import_file
import argparse

QReg_C = import_file("/mnt/t3nfs01/data01/shome/threiten/QReg/dataMC-1/MTR/quantileRegression_chain")


def main(options):

    # QReg_C.setupJoblib('long_6gb',range(options.indWorkerSt,options.indWorkerSt+6))
    if options.year == '2017':
        columns = ['probePt','probeScEta','probePhi','rho','probeEtaWidth','probeSigmaIeIe','probePhiWidth','probeR9','probeS4','probeCovarianceIeIp']
    elif options.year == '2016':
        columns = ['probePt','probeScEta','probePhi','rho','probeEtaWidth','probeSigmaIeIe','probePhiWidth','probeR9','probeS4','probeCovarianceIetaIphi']
    qRC = QReg_C.quantileRegression_chain(options.year,options.EBEE,options.workDir)
    qRC.quantiles = [options.quantile]
    qRC.loadDataDF(options.dataFrame,0,options.n_evts,rsh=False,columns=columns)
    qRC.trainOnData(options.variable,weightsDir=options.weightsDir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-W','--workDir', action='store', type=str,required=True)
    requiredArgs.add_argument('-D','--weightsDir', action='store', type=str, required=True)
    requiredArgs.add_argument('-y','--year', action='store', type=str, required=True)
    requiredArgs.add_argument('-F','--dataFrame', action='store', type=str, required=True)
    requiredArgs.add_argument('-q','--quantile', action='store', type=float, required=True)
    requiredArgs.add_argument('-v','--variable', action='store', type=str, required=True)
    requiredArgs.add_argument('-E','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-N','--n_evts', action='store', type=int, required=True)
    options=parser.parse_args()
    main(options)

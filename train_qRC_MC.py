from import_file import import_file
import argparse

QReg_C = import_file("/mnt/t3nfs01/data01/shome/threiten/QReg/dataMC-1/MTR/quantileRegression_chain")


def main(options):

#    QReg_C.setupJoblib('long_6gb',range(options.indWorkerSt,options.indWorkerSt+6))
    qRC = QReg_C.quantileRegression_chain(options.year,options.EBEE,options.workDir)
    qRC.setupJoblib('long_6gb')
    qRC.loadMCDF(options.dataFrameMC,0,options.n_evts,rsh=False)
    qRC.loadDataDF(options.dataFrameData,0,options.n_evts,rsh=False)
    qRC.trainAllMC(weightsDir=options.weightsDir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-W','--workDir', action='store', type=str,required=True)
    requiredArgs.add_argument('-D','--weightsDir', action='store', type=str, required=True)
    requiredArgs.add_argument('-y','--year', action='store', type=str, required=True)
    requiredArgs.add_argument('-E','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-F','--dataFrameData', action='store', type=str, required=True)
    requiredArgs.add_argument('-M','--dataFrameMC', action='store', type=str, required=True)
    requiredArgs.add_argument('-N','--n_evts', action='store', type=int, required=True)
    options=parser.parse_args()
    main(options)

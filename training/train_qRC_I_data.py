from import_file import import_file
import argparse
import yaml

QReg_I = import_file("/mnt/t3nfs01/data01/shome/threiten/QReg/qRC/quantileRegression_chain_disc")


def main(options):

    stream = file(options.config,'r')
    inp=yaml.load(stream)

    df_name = inp['dataframes']['data_{}'.format(options.EBEE)]
    variables = inp['variables']
    year = str(inp['year'])
    workDir = inp['workDir']
    weightsDir = inp['weightsDir']

    if options.split is not None:
        print 'Using split dfs for training two sets of weights!'
        df_name = df_name + '_spl{}_1M.h5'.format(options.split)
        weightsDir = weightsDir + '/spl{}'.format(options.split)

    
    columns = ['probePt','probeScEta','probePhi','rho'] + variables
    qRC_I = QReg_I.quantileRegression_chain_disc(year,options.EBEE,workDir,variables)
    qRC_I.quantiles = [options.quantile]
    qRC_I.loadDataDF(df_name,0,options.n_evts,rsh=False,columns=columns)
    qRC_I.trainOnData(options.variable,weightsDir=weightsDir)


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

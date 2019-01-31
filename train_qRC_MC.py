from import_file import import_file
import argparse
import yaml

QReg_C = import_file("/mnt/t3nfs01/data01/shome/threiten/QReg/qRC/quantileRegression_chain")

def main(options):

    stream = file(options.config,'r')
    inp=yaml.load(stream)

    df_name_data = inp['dataframes']['data_{}'.format(options.EBEE)]
    df_name_mc = inp['dataframes']['mc_{}'.format(options.EBEE)]
    variables = inp['variables']
    year = str(inp['year'])
    workDir = inp['workDir']
    weightsDir = inp['weightsDir']

    if options.split is not None:
        print 'Using split dfs for training two sets of weights!'
        df_name_data = df_name_data + '_spl{}_1M.h5'.format(options.split)
        df_name_mc = df_name_mc + '_spl{}_1M.h5'.format(options.split)
        weightsDir = weightsDir + '/spl{}'.format(options.split)

    qRC = QReg_C.quantileRegression_chain(year,options.EBEE,workDir,variables)
    if options.backend is not None:
        qRC.setupJoblib(options.backend)
    qRC.loadMCDF(df_name_mc,0,options.n_evts,rsh=False)
    qRC.loadDataDF(df_name_data,0,options.n_evts,rsh=False)
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
    options=parser.parse_args()
    main(options)

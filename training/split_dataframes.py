import argparse
import yaml
import pandas as pd
from sklearn.model_selection import ShuffleSplit

def split_df(df,rsplit=0.5):
    rs = ShuffleSplit(n_splits=1,test_size=rsplit)
    for ind1, ind2 in rs.split(df):
        df_spl1=df.loc[ind1,:]
        df_spl2=df.loc[ind2,:]
    return df_spl1, df_spl2

def main(options):

    stream = file(options.config,'r')
    inp=yaml.load(stream)
    
    df_name_data = inp['dataframes']['data_{}'.format(options.EBEE)]
    df_name_mc = inp['dataframes']['mc_{}'.format(options.EBEE)]
    workDir = inp['workDir']
    
    df_MC = pd.read_hdf('{}/{}.h5'.format(workDir,df_name_mc), stop=options.n_evts)
    df_data = pd.read_hdf('{}/{}.h5'.format(workDir,df_name_data), stop=options.n_evts)

    df_MC1, df_MC2 = split_df(df_MC)
    df_data1, df_data2 = split_df(df_data)

    df_MC1.to_hdf('{}/{}_spl1.h5'.format(workDir,df_name_mc),'df',mode='w',format='t')
    df_MC2.to_hdf('{}/{}_spl2.h5'.format(workDir,df_name_mc),'df',mode='w',format='t')

    df_data1.to_hdf('{}/{}_spl1.h5'.format(workDir,df_name_data),'df',mode='w',format='t')
    df_data2.to_hdf('{}/{}_spl2.h5'.format(workDir,df_name_data),'df',mode='w',format='t')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-E','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-c','--config', action='store', type=str, required=True)
    requiredArgs.add_argument('-N','--n_evts', action='store', type=int, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    options=parser.parse_args()
    main(options)

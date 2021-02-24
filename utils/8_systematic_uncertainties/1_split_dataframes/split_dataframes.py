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

    workDir = options.input_dir
    tmpl = 'df_{}_{}_train'
    tmpl_iso = 'df_{}_{}_Iso_train'

    for dset in ['mc', 'data']:
        for det in ['EB', 'EE']:
            for t in [tmpl, tmpl_iso]:
                name = t.format(dset, det)
                print('Creating dataframe from {}.h5'.format(name))
                inp = pd.read_hdf('{}/{}.h5'.format(workDir, name), stop=options.n_evts)
                df1, df2 = split_df(inp)
                df1.to_hdf('{}/{}_spl1.h5'.format(workDir, name),'df',mode='w',format='t')
                df2.to_hdf('{}/{}_spl2.h5'.format(workDir, name),'df',mode='w',format='t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-N','--n_evts', action='store', type=int, required=True)
    requiredArgs.add_argument('-i','--input_dir', action='store', type=str, required=True)
    options=parser.parse_args()
    main(options)

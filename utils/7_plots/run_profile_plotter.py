import argparse
import itertools
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import root_pandas
from quantile_regression_chain.plotting import parse_yaml as pyml
from quantile_regression_chain.plotting import plot_dmc_hist as pldmc
from quantile_regression_chain.tmva.IdMVAComputer import helpComputeIdMva
from quantile_regression_chain.syst import qRC_systematics as syst
from joblib import delayed, Parallel

def remove_duplicates(vars):
    mask = len(vars) * [True]
    for i in range(len(vars)):
        for j in range(i+1,len(vars)):
            if vars[i] == vars[j]:
                mask[j] = False

    return list(itertools.compress(vars,mask))

def chunked_loader(fpath,columns, **kwargs):
    fitt = pd.read_hdf(fpath, columns=columns, chunksize=10000, iterator=True, **kwargs)
    df = pd.concat(fitt, copy=False)

    return df

def make_final_corr_vars(target_vars):
    corr_var_names = []
    for var in target_vars:
        if 'ID' not in var: var += '_corr_1Reg'
        else: var += 'corrAllFinal'
        corr_var_names.append(var)

    return corr_var_names

def make_corr_vars(target_vars):
    corr_var_names = []
    for var in target_vars:
        if 'ID' not in var: var += '_corr'
        else: var += 'corrAll'
        corr_var_names.append(var)

    return corr_var_names


def check_vars(df, varrs):
    varmiss = len(varrs) * [False]
    for i, var in enumerate(varrs):
        if not var in df.columns:
            varmiss[i] = True

    return varmiss

def main(options):

    base_vars = ['weight', 'probePassEleVeto', 'tagPt', 'tagScEta', 'mass'] #pt already contained in diff var
    diff_vars = ['probePt', 'probeScEta', 'probePhi', 'rho']

    if options.varrs == 'SS':
        #target_vars = ['probeR9', 'probeS4', 'probeCovarianceIeIp', 'probeEtaWidth', 'probePhiWidth', 'probeSigmaIeIe', 'newULPhoID']
        target_vars = ['probeR9', 'probeS4', 'probeCovarianceIeIp', 'probeEtaWidth', 'probePhiWidth', 'probeSigmaIeIe', 'newPhoID']
    elif options.varrs == 'Iso':
        target_vars = ['probeChIso03', 'probeChIso03worst', 'probePhoIso']

    varrs = base_vars + diff_vars + target_vars + make_corr_vars(target_vars)
    if options.final_reg: varrs += make_final_corr_vars(target_vars)
    varrs_data = base_vars + diff_vars + target_vars

    #print varr
    #print varrs_data

    if options.EBEE=='EE': cut_string = 'abs(probeScEta)>1.56 and tagPt>40 and probePt>20 and mass>80 and mass<100 and  probePassEleVeto==0 and abs(tagScEta)<2.5 and abs(probeScEta)<2.5'
    elif options.EBEE=='EB': cut_string = 'abs(probeScEta)<1.4442 and tagPt>40 and probePt>20 and mass>80 and mass<100 and probePassEleVeto==0 and abs(tagScEta)<2.5'
    else: raise NameError('Region has to be either EE or EB')


    if 'probePhoIso03_uncorr' in varrs:
        varrs.pop(varrs.index('probePhoIso03_uncorr'))

    if options.mc.split('.')[-1] == 'root':
        if options.mc_tree is None:
            raise NameError('mc_tree has to be in options if a *.root file is used as input')
        df_mc = root_pandas.read_root(options.mc, options.mc_tree, columns=varrs)
    else:
        #df_mc = pd.read_hdf(options.mc)
        #print df_mc.columns
        df_mc = pd.read_hdf(options.mc, columns=varrs)

    if options.data.split('.')[-1] == 'root':
        if options.data_tree is None:
            raise NameError('data_tree has to be in options if a *.root file is used as input')
        df_data = root_pandas.read_root(options.data, options.data_tree, columns=varrs_data)
    else:
        df_data = pd.read_hdf(options.data, columns=varrs_data+['newPhoID']) #NOTE NOTE: again adjust for inccorect UL ID name in mc)

    print('MC columns: {}'.format(df_mc.columns))
    print('Data columns: {}'.format(df_data.columns))

    #NOTE: can remove all this crap and just chose the EB or EE cut based on the EEEB option in the config
    #NOTE: may have to have a different cut for the actual IDMVA plot though.

    if 'weight_clf' not in df_mc.columns and not options.no_reweight:
        if options.reweight_cut is not None:
            df_mc['weight_clf'] = syst.utils.clf_reweight(df_mc, df_data, n_jobs=10, cut=options.reweight_cut)
        else:
            warnings.warn('Cut from 0th plot used to reweight whole dataset. Make sure this makes sense')
            df_mc['weight_clf'] = syst.utils.clf_reweight(df_mc, df_data, n_jobs=10, cut=cut_string)


    varrs_miss = check_vars(df_mc, varrs)
    varrs_data_miss = check_vars(df_data, varrs_data)
    if any(varrs_miss + varrs_data_miss):
        print('Missing variables from mc df: ', list(itertools.compress(varrs,varrs_miss)))
        print('Missing variables from data df: ', list(itertools.compress(varrs_data,varrs_data_miss)))
        raise KeyError('Variables missing !')

    #use profile plotter class here. NOTE: might have to add some more options in config if constructor needs them
    #                                NOTE: also add corrlabel in config, which will be different for the IDMVA (corrAll)
    for y_var in target_vars:
        for diff_var in diff_vars:
            #plotter = profPlots(df_mc, df_data, 40, 'equ', y_var, diff_var, 'weight_clf', config.EEEB)
            if options.final_reg:
                if 'PhoID' in y_var: plotter = profPlot.profilePlot(df_mc, df_data, 40, options.bintype, y_var, diff_var, 'weight_clf', options.EBEE, corrlabel='corrAll', addlabel='corrAllFinal', addlegd='mc corr final')
                else: plotter = profPlot.profilePlot(df_mc, df_data, 40, options.bintype, y_var, diff_var, 'weight_clf', options.EBEE, corrlabel='_corr', addlabel='_corr_1Reg', addlegd='mc corr final')
            else:
                if 'PhoID' in y_var: plotter = profPlot.profilePlot(df_mc, df_data, 40, options.bintype, y_var, diff_var, 'weight_clf', options.EBEE, corrlabel='corrAll')
                else: plotter = profPlot.profilePlot(df_mc, df_data, 40, options.bintype, y_var, diff_var, 'weight_clf', options.EBEE, corrlabel='_corr')
            plotter.get_quantiles()
            plotter.plot_profile()
            plotter.save(options.outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group()
    requiredArgs.add_argument('-m', '--mc', action='store', type=str, required=True)
    requiredArgs.add_argument('-d', '--data', action='store', type=str, required=True)
    #requiredArgs.add_argument('-c', '--config', action='store', type=str, required=True)
    requiredArgs.add_argument('-o', '--outdir', action='store', type=str, required=True)
    requiredArgs.add_argument('-E', '--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-V', '--varrs', action='store', type=str, required=True)
    optionalArgs = parser.add_argument_group()
    optionalArgs.add_argument('-w', '--no_reweight', action='store_true', default=False)
    optionalArgs.add_argument('-k', '--cutstr', action='store_true', default=False)
    optionalArgs.add_argument('-M', '--recomp_mva', action='store_true', default=False)
    optionalArgs.add_argument('-N', '--n_evts', action='store', type=int)
    optionalArgs.add_argument('-t', '--mc_tree', action='store', type=str)
    optionalArgs.add_argument('-s', '--data_tree', action='store', type=str)
    optionalArgs.add_argument('-u', '--reweight_cut', action='store', type=str)
    optionalArgs.add_argument('-B', '--bintype', action='store', default='equ', type=str)
    optionalArgs.add_argument('-f', '--final_reg', action='store_true', default=False)
    options = parser.parse_args()
    main(options)

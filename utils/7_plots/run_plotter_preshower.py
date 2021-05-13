import argparse
import itertools
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import uproot
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

def make_unique_names(plt_list):

    mtl_list = len(plt_list) * [0]
    for i in range(len(plt_list)):
        mult = 0
        for j in range(i):
            if plt_list[i]['type'] == plt_list[j]['type'] and plt_list[i]['var'] == plt_list[j]['var']:
                mult += 1

        mtl_list[i] = mult

    for i in range(len(plt_list)):
        plt_list[i]['num'] = mtl_list[i]

    return plt_list

def make_vars(plot_dict,extra=[],extens=True):
    ret = []
    for dic in plot_dict:
        ret.append(dic['var'])
        if 'exts' in dic.keys() and extens:
            for ext in dic['exts']:
                ret.append(dic['var'] + ext)

    ret.extend(extra)

    return remove_duplicates(ret)

def check_vars(df, varrs):

    varmiss = len(varrs) * [False]
    for i, var in enumerate(varrs):
        if not var in df.columns:
            varmiss[i] = True

    return varmiss

def main(options):

    # Open mc test dataframe with all variables
    original_mc = pd.read_hdf('/work/gallim/dataframes/UL2018/correct_preshower/full_dataframes_etacut/df_mc_EE_test.h5')

    plot_dict = make_unique_names(pyml.yaml_parser(options.config)())
    varrs = make_vars(plot_dict,['weight', 'probePassEleVeto', 'tagPt', 'tagScEta', 'tagR9', 'probeEtaWidth', 'probePhiWidth','probePhoIso_uncorr', 'probeScEnergy', 'probeSigmaRR'], extens=False)
    print("MC variables {}".format(varrs))

    varrs_data = make_vars(plot_dict,['weight', 'probePassEleVeto', 'tagPt', 'tagScEta', 'tagR9', 'probeEtaWidth', 'probePhiWidth','probePhoIso_uncorr', 'probeScEnergy', 'probeSigmaRR'],extens=False)
    print("Data variables {}".format(varrs_data))

    if 'probePhoIso03_uncorr' in varrs:
        varrs.pop(varrs.index('probePhoIso03_uncorr'))

    if options.mc.split('.')[-1] == 'root':
        flashgg = True
        if options.mc_tree is None:
            raise NameError('mc_tree has to be in options if a *.root file is used as input')
        events = uproot.open("{}:{}".format(options.mc, options.mc_tree))
        df_mc = events.arrays(varrs, library="pd")
    else:
        flashgg = False
        df_mc = pd.read_hdf(options.mc)

    if options.data.split('.')[-1] == 'root':
        if options.data_tree is None:
            raise NameError('data_tree has to be in options if a *.root file is used as input')
        events = uproot.open("{}:{}".format(options.data, options.data_tree))
        df_data = events.arrays(varrs_data, library="pd")
    else:
        df_data = pd.read_hdf(options.data)

    if flashgg:
        df_mc['probePhoIso03_uncorr'] = df_mc['probePhoIso_uncorr']
        df_data['probePhoIso03_uncorr'] = df_data['probePhoIso_uncorr']

    if 'weight_clf' not in df_mc.columns and not options.no_reweight:
    #if not options.no_reweight:
        print('Performing re-weighting')
        if options.reweight_cut is not None:
            df_mc['weight_clf'] = syst.utils.clf_reweight(df_mc, df_data, n_jobs=10, cut=options.reweight_cut)
            df_mc.to_hdf(options.mc, key='df')
        else:
            warnings.warn('Cut for reweighting is taken from 0th and 1st plot. Make sure this is the right one')
            if 'abs(probeScEta)<1.4442' in plot_dict[0]['cut'] and 'abs(probeScEta)>1.56' in plot_dict[1]['cut']:
                df_mc.loc[np.abs(df_mc['probeScEta'])<1.4442,'weight_clf'] = syst.utils.clf_reweight(df_mc.query('abs(probeScEta)<1.4442'), df_data, n_jobs=10, cut=plot_dict[0]['cut'])
                df_mc.loc[np.abs(df_mc['probeScEta'])>1.56,'weight_clf'] = syst.utils.clf_reweight(df_mc.query('abs(probeScEta)>1.56'), df_data, n_jobs=10, cut=plot_dict[1]['cut'])
            else:
                warnings.warn('Cut from 0th plot used to reweight whole dataset. Make sure this makes sense')
                df_mc['weight_clf'] = syst.utils.clf_reweight(df_mc, df_data, n_jobs=10, cut=plot_dict[0]['cut'])

    if flashgg:
        if 'probePhiWidth' in varrs:
            df_data['probePhiWidth'] = df_data['probePhiWidth_Sc']
        if 'probeEtaWidth' in varrs:
            df_data['probeEtaWidth'] = df_data['probeEtaWidth_Sc']
        if 'probePhoIso03' in varrs:
            df_mc['probePhoIso03_uncorr'] = df_mc['probePhoIso_uncorr']

    if options.recomp_mva:
        stride = int(df_mc.index.size/10)

        # Get what is needed
        correctedVariables = ['probeR9', 'probeS4', 'probeCovarianceIeIp', 'probeEtaWidth', 'probePhiWidth', 'probeSigmaIeIe', 'probePhoIso03', 'probeChIso03', 'probeChIso03worst']
        weightsEB = "/work/gallim/weights/id_mva/HggPhoId_94X_barrel_BDT_v2.weights.xml"
        weightsEE = "/work/gallim/weights/id_mva/HggPhoId_94X_endcap_BDT_v2.weights.xml"
        # To remove when not needed
        df_mc['probePhoIso03'] = original_mc['probePhoIso03']
        for var in correctedVariables:
            if var == 'probePhoIso03':
                uncorr = 'probePhoIso_uncorr'
                df_mc['probePhoIso03_uncorr'] = original_mc[uncorr]
            else:
                uncorr = '{}_uncorr'.format(var)
                df_mc[uncorr] = original_mc[uncorr]

        # Data
        print('recomputing photon id for data')
        df_data['probeEnovSCRawEn'] = df_data['phoIdMVA_esEnovSCRawEn']
        df_data['probePhoIdMVA'] = np.concatenate(Parallel(n_jobs=20,verbose=20)(delayed(helpComputeIdMva)(weightsEB,weightsEE,correctedVariables,df_data[ch:ch+stride],'data', False) for ch in range(0,df_data.index.size,stride)))

        # MC uncorrected preshower
        print('Performing MC uncorr preshower')
        df_mc['probeEnovSCRawEn'] = df_mc['phoIdMVA_esEnovSCRawEn']
        df_mc['probePhoIdMVA_uncorr_preshower'] = np.concatenate(Parallel(n_jobs=20,verbose=20)(delayed(helpComputeIdMva)(weightsEB,weightsEE,correctedVariables,df_mc[ch:ch+stride],'data', False) for ch in range(0,df_mc.index.size,stride)))

        # MC corrected preshower
        print('Performing MC corrected preshower')
        df_mc['probeEnovSCRawEn'] = df_mc['phoIdMVA_esEnovSCRawEn_corr_1Reg']
        #df_mc['probeEnovSCRawEn'] = df_mc['phoIdMVA_esEnovSCRawEn_corr']
        df_mc['probePhoIdMVA'] = np.concatenate(Parallel(n_jobs=20,verbose=20)(delayed(helpComputeIdMva)(weightsEB,weightsEE,correctedVariables,df_mc[ch:ch+stride],'data', False) for ch in range(0,df_mc.index.size,stride)))

        # MC corrected flashgg
        df_mc['probePhoIdMVA_corr_flashgg'] = original_mc['probePhoIdMVA']


    varrs_miss = check_vars(df_mc, varrs)
    varrs_data_miss = check_vars(df_data, varrs_data)

    plots = []
    for dic in plot_dict:
        plots.append(pldmc.plot_dmc_hist(df_mc, df_data=df_data, ratio=options.ratio, norm=options.norm, cut_str=options.cutstr, label=options.label, **dic))

    for plot in plots:
        plot.draw()
        plot.save(options.outdir, save_dill=options.save_dill)
        matplotlib.pyplot.close(plot.fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group()
    requiredArgs.add_argument('-m', '--mc', action='store', type=str, required=True)
    requiredArgs.add_argument('-d', '--data', action='store', type=str, required=True)
    requiredArgs.add_argument('-c', '--config', action='store', type=str, required=True)
    requiredArgs.add_argument('-o', '--outdir', action='store', type=str, required=True)
    optionalArgs = parser.add_argument_group()
    optionalArgs.add_argument('-r', '--ratio', action='store_true', default=False)
    optionalArgs.add_argument('-n', '--norm', action='store_true', default=False)
    optionalArgs.add_argument('-p', '--save_dill', action='store_true', default=False)
    optionalArgs.add_argument('-w', '--no_reweight', action='store_true', default=False)
    optionalArgs.add_argument('-k', '--cutstr', action='store_true', default=False)
    optionalArgs.add_argument('-M', '--recomp_mva', action='store_true', default=False)
    optionalArgs.add_argument('-l', '--label', action='store', type=str)
    optionalArgs.add_argument('-N', '--n_evts', action='store', type=int)
    optionalArgs.add_argument('-t', '--mc_tree', action='store', type=str)
    optionalArgs.add_argument('-s', '--data_tree', action='store', type=str)
    optionalArgs.add_argument('-u', '--reweight_cut', action='store', type=str)
    options = parser.parse_args()
    main(options)

import argparse
import pandas as pd
import numpy as np
import warnings
from quantile_regression_chain.plotting import corr_plots
from quantile_regression_chain.syst import qRC_systematics as syst
import os

def main(options):
    base_vars = ['weight', 'probePassEleVeto', 'tagPt', 'tagScEta', 'mass'] #pt already contained in diff var

    variables = ["probePt","probeScEta","probePhi",'rho','probeScEnergy','probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth','probePhoIso','probeChIso03','probeChIso03worst']
    variables_corr  = ["probePt","probeScEta","probePhi",'rho','probeScEnergy','probeCovarianceIeIp_corr','probeS4_corr','probeR9_corr','probePhiWidth_corr','probeSigmaIeIe_corr','probeEtaWidth_corr', 'probePhoIso_corr','probeChIso03_corr','probeChIso03worst_corr']
    if options.final_reg: variables_corr = ["probePt","probeScEta","probePhi",'rho','probeScEnergy','probeCovarianceIeIp_corr_1Reg','probeS4_corr_1Reg','probeR9_corr_1Reg','probePhiWidth_corr_1Reg','probeSigmaIeIe_corr_1Reg','probeEtaWidth_corr_1Reg','probePhoIso_corr_1Reg','probeChIso03_corr_1Reg','probeChIso03worst_corr_1Reg']

    varrs      = variables+base_vars+variables_corr
    varrs_data = variables+base_vars

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
        df_mc = pd.read_hdf(options.mc, columns=varrs)

    if options.data.split('.')[-1] == 'root':
        if options.data_tree is None:
            raise NameError('data_tree has to be in options if a *.root file is used as input')
        df_data = root_pandas.read_root(options.data, options.data_tree, columns=varrs_data)
    else:
        df_data = pd.read_hdf(options.data, columns=varrs_data)

    #add clf reweighting
    if 'weight_clf' not in df_mc.columns and not options.no_reweight:
        if options.reweight_cut is not None:
            df_mc['weight_clf'] = syst.utils.clf_reweight(df_mc, df_data, n_jobs=10, cut=options.reweight_cut)
        else:
            warnings.warn('Cut from 0th plot used to reweight whole dataset. Make sure this makes sense')
            df_mc['weight_clf'] = syst.utils.clf_reweight(df_mc, df_data, n_jobs=10, cut=cut_string)

    df_mc = df_mc.query(cut_string)
    df_data = df_data.query(cut_string)

    corrMatspt = []
    corrMatspt.append(corr_plots.corrMat(df_mc.query('probePt<35'),df_data.query('probePt<35'),variables,variables_corr,'weight_clf','{} qRC probePt<35'.format(options.EBEE)))
    corrMatspt.append(corr_plots.corrMat(df_mc.query('probePt>35 and probePt<50'),df_data.query('probePt>35 and probePt<50'),variables,variables_corr,'weight_clf','{} qRC 35 < probePt < 50'.format(options.EBEE)))
    corrMatspt.append(corr_plots.corrMat(df_mc.query('probePt>50'),df_data.query('probePt>50'),variables,variables_corr,'weight_clf','{} qRC probePt > 50'.format(options.EBEE)))
    for corrMat in corrMatspt:
        corrMat.plot_corr_mat('mc')
        corrMat.plot_corr_mat('mcc')
        corrMat.plot_corr_mat('data')
        corrMat.plot_corr_mat('diff')
        corrMat.plot_corr_mat('diffc')
        #corrMat.save(options.outdir, options.varrs)
        corrMat.save(options.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group()
    requiredArgs.add_argument('-m', '--mc', action='store', type=str, required=True)
    requiredArgs.add_argument('-d', '--data', action='store', type=str, required=True)
    requiredArgs.add_argument('-o', '--outdir', action='store', type=str, required=True)
    requiredArgs.add_argument('-E', '--EBEE', action='store', type=str, required=True)
    optionalArgs = parser.add_argument_group()
    optionalArgs.add_argument('-w', '--no_reweight', action='store_true', default=False)
    optionalArgs.add_argument('-u', '--reweight_cut', action='store', type=str)
    optionalArgs.add_argument('-f', '--final_reg', action='store_true', default=False)
    options = parser.parse_args()
    main(options)

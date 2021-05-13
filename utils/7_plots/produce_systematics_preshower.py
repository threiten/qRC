# Works with ROOT files converted to hdf5 as input

import pandas as pd
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
from quantile_regression_chain.plotting import plot_dmc_hist as pldmc
from quantile_regression_chain.syst import qRC_systematics as syst
import uproot

def plotOrig(systs, df_data, saveDir, cut=None, label=None, zoom=False):

    for i in range(systs.shifts.shape[0]):
        systs.df['probePhoIdMVA_shift{}'.format(i)] = syst.utils.get_quantile(systs.df, df_data, 'probePhoIdMVAtr_shift{}'.format(i),'probePhoIdMVA', weights='weight', inv=True)

    matOrig = systs.df.loc[:,['probePhoIdMVA_shift{}'.format(i) for i in range(systs.shifts.shape[0])]].values
    mini, maxi = syst.utils.findBand(matOrig, np.linspace(-1,1,101), systs.df['weight_clf'])
    mini = mini[10:]
    maxi = maxi[10:]
    # df_data['probePhoIdMVA'] = df_data['probePhoIdMVA']

    dic = {}
    dic['var'] = 'probePhoIdMVA'
    dic['bins'] = 90
    dic['xmin'] = -0.8
    dic['xmax'] = 1
    dic['weightstr_mc'] = 'weight_clf'
    dic['ratio_lim'] = (0.5,1.5) if not zoom else (0.9,1.1)
    dic['type'] = 'dataMC'
    if label is None:
        label = 'IdMVA syst'
    cutBool = False
    if cut is not None:
        dic['cut'] = cut
        cutBool = True
        label = label + ' EB' if 'abs(probeScEta)<1.4442' in cut else label + ' EE'

    plltt = pldmc.plot_dmc_hist(systs.df, df_data=df_data, ratio=True, norm=True, cut_str='', label=label, **dic)
    plltt.draw()
    xc = 0.5*(plltt.bins[1:]+plltt.bins[:-1])
    norm_fac = plltt.data.shape[0] / plltt.mc_weights_cache.sum()
    print(norm_fac)
    plltt.fig.axes[0].fill_between(xc,norm_fac*maxi,norm_fac*mini,color='purple',alpha=0.3,label='Systematic uncertainity',step='mid')
    rdatamc_syst_down = np.divide(plltt.data_hist, norm_fac*mini, dtype=np.float)
    # print(rdatamc_syst_down)
    rdatamc_syst_up = np.divide(plltt.data_hist, norm_fac*maxi)
    rdatamc_syst_down[np.isinf(rdatamc_syst_down)] = 999999
    plltt.fig.axes[1].fill_between(xc,rdatamc_syst_down,rdatamc_syst_up,color='purple',alpha=0.3,label='Systematic uncertainity',step='mid')
    legend = plltt.fig.axes[0].legend()
    if cutBool:
        figsize = plltt.fig.get_size_inches()*plltt.fig.dpi
        pos = legend.get_window_extent()
        ann_pos, lr, tb = plltt.get_annot_pos(pos, figsize)
        lc = {'left': 'left', 'right': 'right'}
        plltt.get_tex_cut()
        plltt.fig.axes[0].annotate(r'\begin{{{0}}}{1}\end{{{0}}}'.format('flush{}'.format(lr), plltt.cut_str_tex), tuple(ann_pos), fontsize=14, xycoords=plltt.fig.axes[0].get_legend(), bbox={'boxstyle': 'square', 'alpha': 0, 'fc': 'w', 'pad': 0}, ha=lr, va=tb)

    plltt.save(saveDir)

def main(options):

    df1 = pd.read_hdf(options.systfile1)
    df2 = pd.read_hdf(options.systfile2)

    columns = ['weight', 'probePt','probeScEta','probePhi','rho','probePhoIdMVA','probeScEta','tagPt','mass','probePassEleVeto','tagScEta']

    if options.infile.split('.')[-1] == 'root' and options.dataFile.split('.')[-1] == 'root':
        flashgg = True
    else:
        flashgg = False

    if flashgg:
        events = uproot.open("{}:{}".format(options.infile, options.mc_tree))
        df_in = events.arrays(columns, library="pd")
    else:
        df_in = pd.read_hdf(options.infile)

    if flashgg:
        events = uproot.open("{}:{}".format(options.dataFile, options.data_tree))
        df_data = events.arrays(columns, library="pd")
    else:
        df_data = pd.read_hdf(options.dataFile)

    print("df1 = {}".format(len(df1.columns)))
    print("df2 = {}".format(len(df2.columns)))
    print("df_data = {}".format(len(df_data.columns)))
    print("df_mc = {}".format(len(df_data.columns)))

    label = 'IdMVA syst'
    if options.cut is not None:
        df_in.query(options.cut, inplace=True, engine='python')
        df_data.query(options.cut, inplace=True, engine='python')
        label = label + ' EB' if 'abs(probeScEta)<1.4442' in options.cut else label + ' EE'

    if 'probePhoIdMVAtr' not in df_in.columns:
        print('Creating probePhoIdMVAtr')
        df_in['probePhoIdMVAtr'] = syst.utils.get_quantile(df_in,df_data,'probePhoIdMVA','probePhoIdMVA',weights='weight')

    #if 'weight_clf' not in df_in.columns or options.forceReweight:
        #print("reweighting input file")
        #df_in['weight_clf'] = syst.utils.clf_reweight(df_in, df_data, n_jobs=40, cut=options.cut)
    if 'weight_clf' not in df_in.columns:
        print('Fetching weight_clf')
        other_df_in = pd.read_hdf('/work/gallim/root_files/tnp_merged_outputs/2018/Preshower/outputMC.h5')
        df_in['weight_clf'] = other_df_in['weight_clf']

    #print("reweighting shift files")
    #df1['weight_clf'] = syst.utils.clf_reweight(df1, df_data, n_jobs=40, cut=options.cut)
    #df2['weight_clf'] = syst.utils.clf_reweight(df2, df_data, n_jobs=40, cut=options.cut)

    df1['newPhoIDtrcorrAll'] = syst.utils.get_quantile(df1,df_data,'newPhoIDcorrAll','probePhoIdMVA',weights='weight')
    df2['newPhoIDtrcorrAll'] = syst.utils.get_quantile(df2,df_data,'newPhoIDcorrAll','probePhoIdMVA',weights='weight')
    df1['newPhoIDtr'] = syst.utils.get_quantile(df1,df_data,'newPhoID','probePhoIdMVA',weights='weight')
    df2['newPhoIDtr'] = syst.utils.get_quantile(df2,df_data,'newPhoID','probePhoIdMVA',weights='weight')
    df_data['probePhoIdMVAtr'] = syst.utils.get_quantile(df_data,df_data,'probePhoIdMVA','probePhoIdMVA',weights='weight')

    shiftFctnDic = {'para': syst.utils.para, 'poly3': syst.utils.poly3, 'poly4': syst.utils.poly4}

    if options.shiftF is not None:
        shiftFctn = shiftFctnDic[options.shiftF]
        label += '_{}'.format(options.shiftF)
    else:
        shiftFctn = None
        label += '_const'

    if options.zoom:
        label += '_zoom'

    shift = syst.systShift(df1, df2, shiftFctn)
    shift.getShiftPars(correctEdges=options.correctEdges)
    if options.plotDir is not None:
        shift.plotFit(options.plotDir, label=label.replace(' ','_'))

    shifts = np.linspace(-1,1,options.nShifts+1)
    shifts = shifts[shifts.nonzero()]

    systs = syst.systematics(df_in, shifts, shift.getShift)
    systs.applShifts(np.sqrt(options.factor))
    cutoff = 0.7 * float(systs.df.index.size)/float(options.nBins)
    systs.getBand(np.linspace(0,1,options.nBins+1),systs.df['weight_clf'],cutoff)

    if options.plotDir is not None and options.dataFile is not None:
        systs.plotBand(df_data, options.plotDir, options.cut, label=label, zoom=options.zoom)
        plotOrig(systs, df_data, options.plotDir, options.cut, label=label, zoom=options.zoom)

    if options.outfile is not None:
        systs.saveSystFile(options.outfile, df_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('required Args')
    requiredArgs.add_argument('-i', '--infile', action='store', type=str, required=True)
    requiredArgs.add_argument('-s', '--systfile1', action='store', type=str, required=True)
    requiredArgs.add_argument('-t', '--systfile2', action='store', type=str, required=True)
    requiredArgs.add_argument('-d', '--dataFile', action='store', type=str, required=True)
    optionalArgs = parser.add_argument_group('Optional Args')
    optionalArgs.add_argument('-n', '--nShifts', action='store', default=20, type=int)
    optionalArgs.add_argument('-b', '--nBins', action='store', default=50, type=int)
    optionalArgs.add_argument('-o', '--outfile', action='store', type=str)
    optionalArgs.add_argument('-c', '--cut', action='store', type=str)
    optionalArgs.add_argument('-p', '--plotDir', action='store', type=str)
    optionalArgs.add_argument('-f', '--factor', action='store', default=1., type=float)
    optionalArgs.add_argument('-F', '--shiftF', action='store', type=str)
    optionalArgs.add_argument('-r', '--forceReweight', action='store_true', default=False)
    optionalArgs.add_argument('-z', '--zoom', action='store_true', default=False)
    optionalArgs.add_argument('-E', '--correctEdges', action='store_true', default=False)
    optionalArgs.add_argument('-dt', '--data_tree', action='store', type=str)
    optionalArgs.add_argument('-mct', '--mc_tree', action='store', type=str)
    options = parser.parse_args()
    main(options)
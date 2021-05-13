from quantile_regression_chain.convert import tree_convert_pkl2xml as pkl2xml
import pickle, gzip


def toxml(var,targetNames,suDir,regName,outDir):
    if var not in regName:
        print('Wrong file selected')
        return
    dic = pickle.load(gzip.open('{}/{}'.format(suDir,regName)))
    if var in dic['Y']:
        clf = dic['clf']
    else:
        print('Wrong file selected')
        return

    N_feat = len(clf.feature_importances_)
    print(N_feat)
    featureNames=['f{}'.format(i) for i in range(N_feat)]

    bdt = pkl2xml.BDTxgboost(clf,featureNames,targetNames)
    fName = regName.replace('.pkl','.xml')
    bdt.to_tmva('{}/{}'.format(outDir,fName))


def main():
    for detector in ['EB', 'EE']:
        for var in ['probeR9','probeS4','probeSigmaIeIe','probeEtaWidth','probePhiWidth','probeCovarianceIeIp']:
            toxml(var,
                    ['{}_corr_diff_scale'.format(var)],
                    '/work/gallim/weights/2018_MIX/good_finals',
                    'weights_finalRegressor_{}_{}.pkl'.format(detector, var),
                    '/work/gallim/weights/2018_MIX/good_finals/xml_files')

        for var in ['probeChIso03','probeChIso03worst']:
            toxml(var,
                    ['{}_corr_diff_scale'.format(var)],
                    '/work/gallim/weights/2018_MIX/good_finals',
                    'weights_finalRegressor_{}_{}.pkl'.format(detector, var),
                    '/work/gallim/weights/2018_MIX/good_finals/xml_files')
            toxml(var,
                    [var],
                    '/work/gallim/weights/2018_MIX/good_finals',
                    'weights_finalTailRegressor_{}_{}.pkl'.format(detector, var),
                    '/work/gallim/weights/2018_MIX/good_finals/xml_files')
            toxml(var,
                    ['peakPeak','peakTail','tailTail'],
                    '/work/gallim/weights/2018_MIX',
                    'data_clf_3Cat_{}_probeChIso03_probeChIso03worst.pkl'.format(detector),
                    '/work/gallim/weights/2018_MIX/good_finals/xml_files')
            toxml(var,
                    ['peakPeak','peakTail','tailTail'],
                    '/work/gallim/weights/2018_MIX',
                    'mc_clf_3Cat_{}_probeChIso03_probeChIso03worst.pkl'.format(detector),
                    '/work/gallim/weights/2018_MIX/good_finals/xml_files')

        for var in ['probePhoIso']:
            toxml(var,
                    ['{}_corr_diff_scale'.format(var)],
                    '/work/gallim/weights/2018_MIX/good_finals',
                    'weights_finalRegressor_{}_{}.pkl'.format(detector, var),
                    '/work/gallim/weights/2018_MIX/good_finals/xml_files')
            toxml(var,
                    [var],
                    '/work/gallim/weights/2018_MIX/good_finals',
                    'weights_finalTailRegressor_{}_{}.pkl'.format(detector, var),
                    '/work/gallim/weights/2018_MIX/good_finals/xml_files')
            toxml(var,
                    ['peak','tail'],
                    '/work/gallim/weights/2018_MIX',
                    'data_clf_p0t_{}_{}.pkl'.format(detector, var),
                    '/work/gallim/weights/2018_MIX/good_finals/xml_files')
            toxml(var,
                    ['peak','tail'],
                    '/work/gallim/weights/2018_MIX',
                    'mc_clf_p0t_{}_{}.pkl'.format(detector, var),
                    '/work/gallim/weights/2018_MIX/good_finals/xml_files')


if __name__ == '__main__':
    main()

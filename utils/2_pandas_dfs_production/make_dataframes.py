import argparse
import yaml
from quantile_regression_chain.quantileRegression_chain import quantileRegression_chain as QReg_C


def parse_arguments():
    parser = argparse.ArgumentParser(
            description = 'Required Arguments')
    parser.add_argument('-D','--sourceDir', action='store', required=True, type = str)
    parser.add_argument('-O','--outDir', action='store', required=True, type = str)
    parser.add_argument('-y','--year', action='store', required=True, type = int)
    parser.add_argument('-E','--EBEE', action='store', required=True, type = str)
    parser.add_argument('-s','--split', action='store', type=float)

    return parser.parse_args()

def main(args):
    sourceDir = args.sourceDir
    outDir = args.outDir
    year = args.year
    EBEE = args.EBEE
    split = args.split

    qRC = QReg_C(year, EBEE, outDir, ['probeR9'])

    qRC.loadROOT(
            '{}/outputMC.root'.format(sourceDir),
            'tagAndProbeDumper/trees/DYJetsToLL_amcatnloFXFX_13TeV_All',
            'df_mc_{}'.format(qRC.EBEE),
            'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeChIso03<6 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0',
            split
            )
    qRC.loadROOT(
            '{}/outputData.root'.format(sourceDir),
            'tagAndProbeDumper/trees/Data_13TeV_All',
            'df_data_{}'.format(qRC.EBEE),
            'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeChIso03<6 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0',
            split
            )

    if EBEE == 'EB':
        qRC.loadROOT(
                '{}/outputMC.root'.format(sourceDir),
                'tagAndProbeDumper/trees/DYJetsToLL_amcatnloFXFX_13TeV_All',
                'df_mc_{}_Iso'.format(qRC.EBEE),
                'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeSigmaIeIe<0.0105 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0',
                split
                )
        qRC.loadROOT(
                '{}/outputData.root'.format(sourceDir),
                'tagAndProbeDumper/trees/Data_13TeV_All',
                'df_data_{}_Iso'.format(qRC.EBEE),
                'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeSigmaIeIe<0.0105 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0',
                split
                )
    elif EBEE == 'EE':
        qRC.loadROOT(
                '{}/outputMC.root'.format(sourceDir),
                'tagAndProbeDumper/trees/DYJetsToLL_amcatnloFXFX_13TeV_All',
                'df_mc_{}_Iso'.format(qRC.EBEE),
                'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeSigmaIeIe<0.028 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0',
                split
                )
        qRC.loadROOT(
                '{}/outputData.root'.format(sourceDir),
                'tagAndProbeDumper/trees/Data_13TeV_All',
                'df_data_{}_Iso'.format(qRC.EBEE),
                'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeSigmaIeIe<0.028 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0',
                split
                )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

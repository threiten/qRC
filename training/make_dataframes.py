annimport argparse
import yaml
import qRC.python.quantileRegression_chain as QReg_C


def main(options):

    qRC = QReg_C.quantileRegression_chain(options.year, options.EBEE, options.outDir, ['probeR9'])
    
    qRC.loadROOT('{}/outputMC.root'.format(options.sourceDir), 'tagAndProbeDumper/trees/DYJetsToLL_amcatnloFXFX_13TeV_All', 'df_mc_{}'.format(qRC.EBEE), 'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeChIso03<6 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0', options.split)
    qRC.loadROOT('{}/outputData.root'.format(options.sourceDir), 'tagAndProbeDumper/trees/Data_13TeV_All', 'df_data_{}'.format(qRC.EBEE), 'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeChIso03<6 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0', options.split)

    if options.EBEE == 'EB':
        qRC.loadROOT('{}/outputMC.root'.format(options.sourceDir), 'tagAndProbeDumper/trees/DYJetsToLL_amcatnloFXFX_13TeV_All', 'df_mc_{}_Iso'.format(qRC.EBEE), 'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeSigmaIeIe<0.0105 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0', options.split)
        qRC.loadROOT('{}/outputData.root'.format(options.sourceDir), 'tagAndProbeDumper/trees/Data_13TeV_All', 'df_data_{}_Iso'.format(qRC.EBEE), 'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeSigmaIeIe<0.0105 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0', options.split)
    elif options.EBEE == 'EE':
        qRC.loadROOT('{}/outputMC.root'.format(options.sourceDir), 'tagAndProbeDumper/trees/DYJetsToLL_amcatnloFXFX_13TeV_All', 'df_mc_{}_Iso'.format(qRC.EBEE), 'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeSigmaIeIe<0.028 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0', options.split)
        qRC.loadROOT('{}/outputData.root'.format(options.sourceDir), 'tagAndProbeDumper/trees/Data_13TeV_All', 'df_data_{}_Iso'.format(qRC.EBEE), 'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeSigmaIeIe<0.028 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0', options.split)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-D','--sourceDir', action='store', required=True)
    requiredArgs.add_argument('-O','--outDir', action='store', required=True)
    requiredArgs.add_argument('-y','--year', action='store', required=True)
    requiredArgs.add_argument('-E','--EBEE', action='store', required=True)
    requiredArgs.add_argument('-s','--split', action='store', type=float, required=True)
    # optionalArgs = parser.add_argument_group('Optional Arguements')
    # optionalArgs.add_argument('-l', '--label', action='store', default='')
    options=parser.parse_args()
    main(options)

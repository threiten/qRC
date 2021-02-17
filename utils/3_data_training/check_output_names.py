import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(
            description = 'Print names of missing data regressors')

    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        type=str,
        help="Path to weights directory")

    return parser.parse_args()

def main(args):
    directory = args.directory

    # Supposed output
    detectors = ['EB', 'EE']
    variables = ["probeCovarianceIeIp", "probeSigmaIeIe", "probeEtaWidth", "probePhiWidth", "probeR9", "probeS4", "probePhoIso", "probeChIso03", "probeChIso03worst"]
    variables_iso = ["probePhoIso", "probeChIso03", "probeChIso03worst"]
    quantiles = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]

    data_outputs = []
    for detector in detectors:
        for variable in variables:
            for quantile in quantiles:
                name = '_'.join(['data', 'weights', detector, variable, str(quantile).replace('.', 'p')]) + '.pkl'
                data_outputs.append(name)
    data_outputs = data_outputs + ['data_clf_3Cat_EB_probeChIso03_probeChIso03worst.pkl', 'data_clf_3Cat_EE_probeChIso03_probeChIso03worst.pkl', 'data_clf_p0t_EB_probePhoIso.pkl', 'data_clf_p0t_EE_probePhoIso.pkl']

    # Check
    data_missing = []
    for name in data_outputs:
        if name not in os.listdir(directory):
            data_missing.append(name)
    for name in data_missing:
        print(name)
    print('Missing {} data models'.format(len(data_missing)))

if __name__ == '__main__':
    args = parse_arguments()
    main(args)

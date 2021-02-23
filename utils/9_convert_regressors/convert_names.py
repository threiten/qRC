import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert TnP-specific variables naming in xml regressors to Flashgg convention")

    parser.add_argument(
        "--directory",
        required=True,
        type=str,
        help="Directory where the .xml files are stored")

    return parser.parse_args()

def change_name(file_name, naming):
    new_name = file_name
    for tnp, fgg in naming.items():
        new_name = new_name.replace(tnp, fgg)
    return new_name

def main(args):
    source_dir = args.directory

    naming = {
            'probeChIso03': 'chIso',
            'worst': 'Worst',
            'probePhoIso': 'phoIso',
            'probeCovarianceIeIp': 'sieip',
            'probeSigmaIeIe': 'sieie',
            'probeEtaWidth': 'etaWidth',
            'probePhiWidth': 'phiWidth',
            'probeR9': 'r9',
            'probeS4': 's4',

            'p0t': 'p2t'
            }

    for file_name in os.listdir(source_dir):
        os.rename('{}/{}'.format(source_dir, file_name),
                '{}/{}'.format(source_dir, change_name(file_name, naming)))



if __name__ == "__main__":
    args = parse_arguments()
    main(args)

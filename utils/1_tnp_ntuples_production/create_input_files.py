"""
Create unified data and montecarlo ROOT files from flashgg TagAndProbe output NTuples.

Run with:

$ python --config_file config.yml

where config.yml has the following structure:

data:
  base_dir:
    "/work/gallim/root_files/tnp_original/test_data_UL18"
  file_tmpl:
    "output_EGamma_alesauva-UL2018_0-10_6_4-v0-Run2018{}-12Nov2019_UL2018-{}-981b04a73c9458401b9ffd78fdd24189_USER_{}.root"
  number:
    500
  tree_path:
    "tagAndProbeDumper/trees/Data_13TeV_All"
  output_file:
    "/work/gallim/root_files/tnp_merged_outputs/2018/final_test/outputData.root"
  runs_id:
    A: "v2"
    B: "v2"
    C: "v2"
    D: "v4"

mc:
  base_dir:
    "/work/gallim/root_files/tnp_original/test_mc_UL18"
  file_tmpl:
    "output_DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_alesauva-UL2018_0-10_6_4-v0-RunIISummer19UL18MiniAOD-106X_upgrade2018_realistic_v11_L1v1-v2-b5e482a1b1e11b6e5da123f4bf46db27_USER_{}.root"
  number:
    500
  tree_path:
    "tagAndProbeDumper/trees/DYJetsToLL_amcatnloFXFX_13TeV_All"
  output_file:
    "/work/gallim/root_files/tnp_merged_outputs/2018/final_test/outputMC.root"

"""
import argparse
import yaml
import ROOT


def parse_arguments():
    parser = argparse.ArgumentParser(
            description = "Create unified output for TnP")

    parser.add_argument(
        "-c",
        "--config_file",
        required=True,
        type=str,
        help="Path to YAML config file")

    return parser.parse_args()


def fill_tchain(base_dir, file_name, number, tree_path, runs_id = None):
    if runs_id is None:
        runs_id = []

    chain = ROOT.TChain()
    if runs_id:
        for run, idd in runs_id.items():
            for num in range(number):
                chain.Add(base_dir + '/' + file_name.format(run, idd, num) + '/' + tree_path)
    else:
        for num in range(1, number):
            chain.Add(base_dir + '/' + file_name.format(num) + '/' + tree_path)

    return chain


def dump_snapshot(chain, output_file, output_tree_name, variables = None):
    rdf = ROOT.RDataFrame(chain)
    f = ROOT.TFile(output_file, "RECREATE")
    if variables:
        rdf.Snapshot(output_tree_name, output_file, variables)
    else:
        rdf.Snapshot(output_tree_name, output_file)
    f.Close()

    return rdf


def main(args):

    config_file = args.config_file

    with open(config_file, "r") as stream:
        input_dict = yaml.safe_load(stream)

        # Create and fill TChain
        data_chain = fill_tchain(
                input_dict["data"]["base_dir"],
                input_dict["data"]["file_tmpl"],
                input_dict["data"]["number"],
                input_dict["data"]["tree_path"],
                input_dict["data"]["runs_id"]
                )

        # Create RDataFrame and dump
        print("Dumping data")
        data_rdf = dump_snapshot(data_chain, input_dict["data"]["output_file"], input_dict["data"]["tree_path"])

        # Create and fill TChain
        mc_chain = fill_tchain(
                input_dict["mc"]["base_dir"],
                input_dict["mc"]["file_tmpl"],
                input_dict["mc"]["number"],
                input_dict["mc"]["tree_path"]
                )

        # Create RDataFrame and dump
        print("Dumping mc")
        mc_rdf = dump_snapshot(mc_chain, input_dict["mc"]["output_file"], input_dict["mc"]["tree_path"])


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

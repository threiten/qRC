import argparse
import re
import itertools
from statistics import mean
import os

def parse_arguments():
    parser = argparse.ArgumentParser(
    description="Get iteration with best parameters")

    parser.add_argument(
        "--directory",
        required=True,
        type=str,
        help="Directory containing .txt files")

    return parser.parse_args()



def print_best_parameters(input_file):

    print("File {}".format(input_file))

    with open(input_file) as f:
        lines = f.readlines()

    # Make iterations dictionary
    iterations = {}
    counter = 1

    # Get important lines from file
    for line in lines:
        if line.startswith("Training sample"):
            iterations[counter] = {}
            iterations[counter]["training_sample_line"] = line
            iterations[counter]["test_score_line"] = lines[lines.index(line) + 1]
            counter += 1

    # Get training and target values
    for num, dic in iterations.items():
        s = re.search(r'\[(.*)\]', dic["training_sample_line"]).group()
        s = s.strip("[")
        s = s.strip("]")
        s = s.split()
        dic["training_sample_values"] = [float(n) for n in s]
        # Check that training scores are not too distant from each other
        for a, b in itertools.combinations(dic["training_sample_values"], 2):
            if abs(a - b) > 0.05:
                print("Too much difference in training scores from iteration {}".format(num))
        # Avg of training sample values
        dic["training_avg_value"] = mean(dic["training_sample_values"])
        # Test score value
        # Difference in characters between colors and uncolored lines
        if dic["test_score_line"].startswith("| \x1b[0m"):
            s = dic["test_score_line"].strip("| \x1b[0m")
            s = s.strip("\x1b[0m |\n")
            s = s.split("\x1b[0m | \x1b[0m")
        elif dic["test_score_line"].startswith("| \x1b[95m"):
            s = dic["test_score_line"].strip("| \x1b[9")
            s = s.strip("\x1b[0m |\n")
            s = s.split("\x1b[0m | \x1b[95m")
        else:
            print("Something went wrong")
        dic["test_value"] = float(s[1])

        # Difference between target value and average of training samples values
        # Should be less than 0.1
        dic["diff"] = abs(dic["training_avg_value"] - dic["test_value"])

        #print("{}: avg training score: {}, target: {}, diff: {}".format(num, dic["training_avg_value"], dic["test_value"], dic["diff"]))

    # Compute scores
    test_values = {}
    diffs = {}
    weights = {}
    for n, dic in iterations.items():
        test_values[n] = dic["test_value"]
        diffs[n] = dic["diff"]
        weights[n] = [0.0, 0.0]

    test_values = {k: v for k, v in sorted(test_values.items(), key=lambda item: item[1], reverse=True)}
    print("\t test score order: {}".format(list(test_values.keys())))
    diffs = {k: v for k, v in sorted(diffs.items(), key=lambda item: item[1])}

    final_scores = {}
    for i in list(range(1, len(iterations) + 1)):
        d = diffs[i]
        tv = test_values[i]
        if d > 0.099:
            weights[i] = [0.8, 0.2]
        elif 0.049 < d < 0.099:
            weights[i] = [0.3, 0.7]
        elif d < 0.049:
            weights[i] = [0.1, 0.9]
        d_score = weights[i][0]*(len(diffs) - list(diffs.values()).index(d))
        tv_score = weights[i][1]*(len(test_values) - list(test_values.values()).index(tv))
        final_scores[i] = tv_score + d_score

    max_place = -1
    max_score = 0
    for n, score in final_scores.items():
        if score > max_score:
            max_place = n
            max_score = score
        #print("{}: final score: {}".format(n, dic["final_score"]))
    print("\t best value found in {}".format(max_place))


def main(args):

    directory = args.directory

    for f in os.listdir(directory):
        if f.endswith(".txt"):
            print_best_parameters(f)



if __name__ == "__main__":
    args = parse_arguments()
    main(args)

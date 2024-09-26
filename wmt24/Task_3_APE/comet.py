import argparse
import subprocess
    
# Setting up argument parser
parser = argparse.ArgumentParser(description="Compute COMET scores for given source, hypothesis, and reference files.")
parser.add_argument('--src', type=str, help="Path to the source text file.")
parser.add_argument('--pred', type=str, help="Path to the hypothesis text file.")
parser.add_argument('--ref', type=str, help="Path to the reference text file.")

# Parsing arguments
args = parser.parse_args()


command = "comet-score -s " + args.ref + " -t " + args.pred + " -r " + args.ref
process = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
print("COMET Scores:", process.stdout)
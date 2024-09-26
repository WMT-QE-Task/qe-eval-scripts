import argparse
import subprocess
import os
from indicnlp.tokenize.indic_tokenize import trivial_tokenize


def read_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [sentence.strip() for sentence in file.readlines()]


def tokenize_data(sentences, type='hypothesis'):
    tok_sents = []
    for line in sentences:
        tok_pred = " ".join(trivial_tokenize(line))
        tok_sents.append(tok_pred)
    
    if type == "hypothesis":
        with open('hypothesis.txt', 'w') as f:
            for line in tok_sents:
                f.write(str(line).strip() + '\n')
    else:
        with open('reference.txt', 'w') as f:
            for line in tok_sents:
                f.write(str(line).strip() + '\n')


def compute_bleu_ter():
    # Command: BLEU score
    command_bleu = "sacrebleu reference.txt -i hypothesis.txt -m bleu -w 4 -b --tokenize none"

    # Command: CHRF score
    command_chrf = "sacrebleu reference.txt -i hypothesis.txt -m chrf -w 4 -b --tokenize none"

    # Run the BLEU command
    process_bleu = subprocess.run(command_bleu, shell=True, check=True, text=True, capture_output=True)
    print("BLEU Score:", process_bleu.stdout)

    # Run the CHRF command
    process_chrf = subprocess.run(command_chrf, shell=True, check=True, text=True, capture_output=True)
    print("CHRF Score:", process_chrf.stdout)

    os.remove('hypothesis.txt')
    os.remove('reference.txt')


# Setting up argument parser
parser = argparse.ArgumentParser(description="Compute BLEU and CHRF scores for given hypothesis, and reference files.")
parser.add_argument('--pred', type=str, help="Path to the hypothesis text file.")
parser.add_argument('--ref', type=str, help="Path to the reference text file.")

# Parsing arguments
args = parser.parse_args()

# Perform Scoring
tokenize_data(read_sentences(args.pred))
tokenize_data(read_sentences(args.ref), type='reference')
compute_bleu_ter()
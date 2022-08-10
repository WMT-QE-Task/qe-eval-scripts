#!/usr/bin/env python
import sys
import os
import os.path
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr, spearmanr


def read_sentence_data(gold_sent_fh, model_sent_fh):
    gold_scores = [float(line.strip()) for line in gold_sent_fh]
    model_scores = [float(line.strip()) for line in model_sent_fh]
    assert len(gold_scores) == len(model_scores)
    return gold_scores, model_scores


def read_word_data(gold_explanations_fh, model_explanations_fh):
    def ok_bad_to_int(s):
        mapping = {'OK': 0, 'BAD': 1}
        assert s.strip() in mapping:
        return mapping[s.strip()]

    gold_explanations = [list(map(ok_bad_to_int, line.split())) for line in gold_explanations_fh]
    model_explanations = [list(map(float, line.split())) for line in model_explanations_fh]
    assert len(gold_explanations) == len(model_explanations)
    for i in range(len(gold_explanations)):
        assert len(gold_explanations[i]) == len(model_explanations[i])
        assert len(gold_explanations[i]) > 0
    return gold_explanations, model_explanations


def validate_word_level_data(gold_explanations, model_explanations):
    valid_gold, valid_model = [], []
    for gold_expl, model_expl in zip(gold_explanations, model_explanations):
        if sum(gold_expl) == 0 or sum(gold_expl) == len(gold_expl):
            continue
        else:
            valid_gold.append(gold_expl)
            valid_model.append(model_expl)
    return valid_gold, valid_model


def compute_auc_score(gold_explanations, model_explanations):
    res = 0
    for i in range(len(gold_explanations)):
        res += roc_auc_score(gold_explanations[i], model_explanations[i])
    return res / len(gold_explanations)


def compute_ap_score(gold_explanations, model_explanations):
    res = 0
    for i in range(len(gold_explanations)):
        res += average_precision_score(gold_explanations[i], model_explanations[i])
    return res / len(gold_explanations)


def compute_rec_topk(gold_explanations, model_explanations):
    res = 0
    for i in range(len(gold_explanations)):
        idxs = np.argsort(model_explanations[i])[::-1][:sum(gold_explanations[i])]
        res += len([idx for idx in idxs if gold_explanations[i][idx] == 1])/sum(gold_explanations[i])
    return res / len(gold_explanations)


def evaluate_word_level(gold_explanations, model_explanations):
    gold_explanations, model_explanations = validate_word_level_data(gold_explanations, model_explanations)
    auc_score = compute_auc_score(gold_explanations, model_explanations)
    ap_score = compute_ap_score(gold_explanations, model_explanations)
    rec_topk = compute_rec_topk(gold_explanations, model_explanations)
    print('Recall at top-K: {:.3f}'.format(rec_topk))
    print('AUC score: {:.3f}'.format(auc_score))
    print('AP score: {:.3f}'.format(ap_score))
    return auc_score, ap_score, rec_topk


def evaluate_sentence_level(gold_scores, model_scores):
    pear = pearsonr(gold_scores, model_scores)[0]
    spea = spearmanr(gold_scores, model_scores)[0]
    print('Pearson correlation: {:.3f}'.format(pear))
    print('Spearman correlation: {:.3f}'.format(spea))
    return pear, spea

def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Default scores
        TGT_AUC, TGT_AP, TGT_RECK = -1.0, -1.0, -1.0
        SEN_PEAR = -2.0
        SEN_SPEA = -2.0

        # Open output file
        output_filename = os.path.join(output_dir, 'scores.txt')
        output_file = open(output_filename, 'w')

        # Read submission metadata
        submission_metadata = open(os.path.join(submit_dir, "metadata.txt"), "r").readlines()
        submission_metadata = [l for l in submission_metadata if l.strip() != '']
        assert len(submission_metadata) == 2, "The metadata.txt must have exactly two non-empty lines which are for the teamname and the short system description, respectively."
        teamname = submission_metadata[0].strip()
        assert teamname != '', "The first line of metadata.txt must contain your team name. You might use your CodaLab username as your teamname."
        description = submission_metadata[1].strip()
        assert description != '', "The second line of metadata.txt must contain a short description (2-3 sentences) of the system you used to generate the results. This description will not be shown to other participants."
        print("----- Metadata -----")
        # print("Language pair: "+str(gold_metadata['lang_pair']))
        print("Teamname: "+str(teamname))
        print("Description: "+str(description))

        # Open gold labels
        gold_scores_fh = open(os.path.join(truth_dir, "sentence.gold"), "r")
        gold_target_expl_fh = open(os.path.join(truth_dir, "target.gold"), "r")

        # Process submissions
        # 1. Sentence-level correlations
        print("----- Sentence-level evaluation -----")
        submission_scores_fh = open(os.path.join(submit_dir, "sentence.submission"), "r")
        gold_scores, model_scores = read_sentence_data(gold_scores_fh, submission_scores_fh)
        SEN_PEAR, SEN_SPEA = evaluate_sentence_level(gold_scores, model_scores)

        # 2. Target explanations
        print("----- Target explanation evaluation -----")
        submission_target_expl_fh = open(os.path.join(submit_dir, "target.submission"), "r")
        gold_target_explanations, model_target_explanations = read_word_data(gold_target_expl_fh, submission_target_expl_fh)
        TGT_AUC, TGT_AP, TGT_RECK = evaluate_word_level(gold_target_explanations, model_target_explanations)

        # Write output file
        output_file.write("TGT_RECK:" + str(TGT_RECK) + "\n")
        output_file.write("TGT_AUC:" + str(TGT_AUC) + "\n")
        output_file.write("TGT_AP:" + str(TGT_AP) + "\n")
        output_file.write("SEN_PEAR:" + str(SEN_PEAR) + "\n")
        output_file.write("SEN_SPEA:" + str(SEN_SPEA))
        output_file.close()

if __name__ == '__main__':
    main()

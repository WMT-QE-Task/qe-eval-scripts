#!/usr/bin/env python
import sys
import os
import argparse
import codecs
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from collections import defaultdict

"""
Scoring programme for WMT'23 Task 1 Sentence-level
"""

language_pairs = ['en-de', 'zh-en', 'he-en', 'en-mr', 'en-hi', 'en-gu', 'en-te', 'en-ta']

def parse_submission(pred_list, goldlabel_bool, hallucinations):
    disk_footprint = 0
    model_params = 0
    ensemble = 1
    lp_dict = defaultdict(list)

    if not goldlabel_bool:
        disk_footprint = pred_list[0].rstrip()
        model_params = pred_list[1].rstrip()
        ensemble = pred_list[2].rstrip()
        pred_list = pred_list[3:]

    else:
        # account for header
        pred_list = pred_list[1:]
    for line in pred_list:
        pred = line.strip().split('\t')
        assert len(pred) == 4, \
                "Incorrect format, expecting (tab separated): <LP> <METHOD_NAME> <SEGMENT_NUMBER> <SEGMENT_SCORE>."

        lp_str = pred[0]
        lp_dict[lp_str.lower()].append(pred[1:])

    lp_dict_keys = list(lp_dict.keys())
    
    for lp_str in lp_dict_keys:
        lp_segments = lp_dict[lp_str]

        tmp_lp_segments = {}
        for _, seg_nb, seg_score in lp_segments:
            hall_idx =  hallucinations[lp_str]
            if goldlabel_bool and str(seg_nb) in hall_idx:
                tmp_lp_segments[seg_nb] = -100.00
            else:
                tmp_lp_segments[seg_nb] = float(seg_score)
        lp_dict[lp_str] = [tmp_lp_segments[str(i)] for i in range(len(tmp_lp_segments))]


    return disk_footprint, model_params, ensemble, lp_dict

def parse_hallucinations(hal_list):
    lp_dict = defaultdict(list)

    for line in hal_list[1:]:
        lp_idx = line.strip().split('\t')
        assert len(lp_idx) == 2, \
                "Incorrect format, expecting (tab separated): <LP> <SEGMENT_NUMBER> ."

        lp_str = lp_idx[0]
        lp_dict[lp_str.lower()].append(lp_idx[1])

    return lp_dict

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('system', help='System file')
    # parser.add_argument('gold', help='Gold output file')
    # parser.add_argument('hallucinations', help='Hallucination index file')
    # parser.add_argument('--baseline', help='Baseline predictions file')
    # parser.add_argument('--outfile', help='Scores.txt  file')
    # args = parser.parse_args()

    # as per the metadata file, input and output directories are the arguments

    [_, input_dir, output_dir] = sys.argv
    reference_dir = os.path.join(input_dir, 'ref')
    submission_dir = os.path.join(input_dir, 'res')

    goldlabels_file_name= 'gold_labels_task1_sentence.tsv'
    baseline_file_name= ''
    hallucinations_file_name= 'gold_hallucinations_task1_sentence.tsv'
    submission_file_name = "predictions.txt"

    if baseline_file_name!='':
        compute_baseline=True
    else:
        compute_baseline=False

    goldlabels_path = os.path.join(reference_dir, goldlabels_file_name)
    if compute_baseline:
        baseline_path = os.path.join(reference_dir, baseline_file_name)
    hallucinations_path = os.path.join(reference_dir, hallucinations_file_name)
    submission_path = os.path.join(submission_dir, submission_file_name)

    print("Loading hallucination indices...")
    with codecs.open(hallucinations_path, 'r', encoding='utf-8') as hal_file:
        hal_idx = parse_hallucinations(hal_file.readlines())
    print("done.")
    
    if compute_baseline:
        print("Loading baseline...")
        with codecs.open(baseline_path, 'r', encoding='utf-8') as base_file:
            _, _, _, baseline = parse_submission(base_file.readlines(), False, hal_idx)
        print("done.")
        
    print("Loading goldlabels...")
    # with codecs.open(args.gold, 'r', encoding='utf-8') as truth_file:
    with codecs.open(goldlabels_path, 'r', encoding='utf-8') as truth_file:
        _, _, _, goldlabels = parse_submission(truth_file.readlines(), True, hal_idx)
    print("done.")

    print("Loading your predictions...")
    with codecs.open(submission_path, 'r', encoding='utf-8') as submission_file:
        disk_footprint, model_params, ensemble, predictions = parse_submission(submission_file.readlines(), False, hal_idx)
    print("done.")

    all_scores = []
    final_score_lines = []
    print("Computing scores...")
    final_dict = {}
    baseline_dict = {}
    
    if compute_baseline:
        for lp_str in baseline:
            tmp_base = {}
            tmp_gold = {}
            
            hall_indices = [eval(i) for i in hal_idx[lp_str]]
            clean_gold = np.delete(goldlabels[lp_str], hall_indices)
            clean_base = np.delete(baseline[lp_str], hall_indices)   
            
            spearman = spearmanr(clean_gold, clean_base)[0]
            pearson = pearsonr(clean_gold, clean_base)[0]
            kendall = kendalltau(clean_gold, clean_base)[0]
            baseline_dict[lp_str+'_pearson'] = pearson
            baseline_dict[lp_str+'_spearman'] = spearman
            baseline_dict[lp_str+'_kendall'] = kendall
        
    for lp_str in predictions:
        print("\t for {}...".format(lp_str))
        print("\t for {}...".format(lp_str), file=sys.stderr)
        assert len(predictions[lp_str]) == len(goldlabels[lp_str]), \
                "Incorrect number of predicted scores for {}, expecting {}, given {}.".format(
                        lp_str, len(goldlabels[lp_str]), len(predictions[lp_str])
                       )
        tmp_preds = {}
        tmp_gold = {}
        
        hall_indices = [eval(i) for i in hal_idx[lp_str]]
        clean_gold = np.delete(goldlabels[lp_str], hall_indices)
        clean_pred = np.delete(predictions[lp_str], hall_indices)    
        
        spearman = spearmanr(clean_gold, clean_pred)[0]
        pearson = pearsonr(clean_gold, clean_pred)[0]
        kendall = kendalltau(clean_gold, clean_pred)[0]
        final_dict[lp_str+'_pearson'] = pearson
        final_dict[lp_str+'_spearman'] = spearman
        final_dict[lp_str+'_kendall'] = kendall
        all_scores.append([spearman, pearson, kendall])
        if compute_baseline:
            base_s = baseline_dict[lp_str+'_spearman']
            base_p = baseline_dict[lp_str+'_pearson']
            base_k = baseline_dict[lp_str+'_kendall']
        # final_score_lines.append(lp_str.upper()+'\n')
        
        final_score_lines.append(lp_str.replace('-','')+"_spearman: {:.4}".format(spearman))
        print(lp_str.replace('-','')+"_spearman: {:.4}".format(spearman))
        print(lp_str.replace('-','')+"_spearman: {:.4}".format(spearman), file=sys.stderr)
        final_score_lines.append(lp_str.replace('-','')+"_pearson: {:.4}".format(pearson))
        print(lp_str.replace('-','')+"_pearson: {:.4}".format(pearson))
        print(lp_str.replace('-','')+"_pearson: {:.4}".format(pearson), file=sys.stderr)
        final_score_lines.append(lp_str.replace('-','')+"_kendall: {:.4}".format(kendall))
        print(lp_str.replace('-','')+"_kendall: {:.4}".format(kendall))
        print(lp_str.replace('-','')+"_kendall: {:.4}".format(kendall), file=sys.stderr)
        # final_score_lines.append('\n')

        if compute_baseline:
            if spearman >= base_s:
                print('Above baseline score for Spearman (primary metric) by: {:.4}'.format(spearman - base_s))
                print('Above baseline score for Spearman (primary metric) by: {:.4}'.format(spearman - base_s), file=sys.stderr)
            else:
                print('Below baseline score for Spearman (primary metric) by: {:.4}'.format(spearman - base_s))
                print('Below baseline score for Spearman (primary metric) by: {:.4}'.format(spearman - base_s), file=sys.stderr)
        print('-----')
        print('-----', file=sys.stderr)
    
    
    # print("\t averaging...")
    average_scores = np.mean(np.array(all_scores), axis=0)
    # print("done.")

    # final_score_lines.append('\n---------- MULTILINGUAL PERFORMANCE ------------\n')
    if set(goldlabels.keys())==set(predictions.keys()):
        final_dict['avg_pearson'] = average_scores[0]
        final_dict['avg_spearman'] = average_scores[1]
        final_dict['avg_kendall'] = average_scores[2]
        final_score_lines.append("avg_spearman: {:.4}".format(average_scores[0]))
        print("avg_spearman: {:.4}".format(average_scores[0]))
        print("avg_spearman: {:.4}".format(average_scores[0]), file=sys.stderr)
        final_score_lines.append("avg_pearson: {:.4}".format(average_scores[1]))
        print("avg_pearson: {:.4}".format(average_scores[1]))
        print("avg_pearson: {:.4}".format(average_scores[1]), file=sys.stderr)
        final_score_lines.append("avg_kendall: {:.4}".format(average_scores[2]))
        print("avg_kendall: {:.4}".format(average_scores[2]))
        print("avg_kendall: {:.4}".format(average_scores[2]), file=sys.stderr)
    else:
        final_score_lines.append("avg_spearman: n/a")
        final_score_lines.append("avg_pearson: n/a")
        final_score_lines.append("avg_kendall: n/a")
        print("avg_spearman: n/a")
        print("avg_spearman: n/a", file=sys.stderr)
        print("avg_pearson: n/a")
        print("avg_pearson: n/a", file=sys.stderr)
        print("avg_kendall: n/a")
        print("avg_kendall: n/a", file=sys.stderr)

    print('\n------------------------------------------------\n')
    print('\n------------------------------------------------\n', file=sys.stderr)
    final_score_lines.append("ensembles: {}".format(ensemble))
    print("ensembles: {}".format(ensemble))
    print("ensembles: {}".format(ensemble), file=sys.stderr)
    final_score_lines.append("model_params: {}".format(model_params))
    print("model_params: {}".format(model_params))
    print("model_params: {}".format(model_params), file=sys.stderr)
    final_score_lines.append("disk footprint: {}".format(disk_footprint))
    print("disk footprint: {}".format(disk_footprint))
    print("disk footprint: {}".format(disk_footprint), file=sys.stderr)


    # with open(args.outfile, 'w', encoding='utf-8') as wf:
    with open(os.path.join(output_dir, 'scores.txt'), 'w', encoding='utf-8') as wf:
        for line in final_score_lines:
            wf.write(line+'\n')


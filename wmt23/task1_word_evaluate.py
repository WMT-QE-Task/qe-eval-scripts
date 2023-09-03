from __future__ import division, print_function

import os, sys
import numpy as np
# from argparse import ArgumentParser
from sklearn.metrics import f1_score, matthews_corrcoef
import codecs

import argparse
from collections import defaultdict

"""
Scoring programme for WMT'23 Task 1 **word-level**
"""

# -------------PREPROCESSING----------------
def list_of_lists(a_list):
    '''
    check if <a_list> is a list of lists
    '''
    if isinstance(a_list, (list, tuple, np.ndarray)) and \
            len(a_list) > 0 and \
            all([isinstance(l, (list, tuple, np.ndarray)) for l in a_list]):
        return True
    return False


def check_word_tag(words_seq, tags_seq, dataset_name='', gap_tags=False):
    '''
    check that two lists of sequences have the same number of elements
    :gaps: --- checking number of gaps between words (has to be words+1)
    '''
    for idx, (words, tags) in enumerate(zip(words_seq, tags_seq)):
        if gap_tags:
            words.append(' ')
        assert(len(words) == len(tags)), "Numbers of words and tags don't match in sequence %d of %s: %i and %i" % (
            idx,
            dataset_name,
            len(words),
            len(tags)
        )


def check_words(ref_words, pred_words):
    '''
    check that words in reference and prediction match
    '''
    assert(len(ref_words) == len(pred_words)), \
        "Number of word sequences doesn't match in reference and hypothesis: %d and %d" % (len(ref_words),
                                                                                           len(pred_words))
    for idx, (ref, pred) in enumerate(zip(ref_words, pred_words)):
        ref_str = ' '.join(ref).lower()
        pred_str = ' '.join(pred).lower()
        assert(ref_str == pred_str), \
            "Word sequences don't match in reference and hypothesis at line %d:\n\t%s\n\t%s\n" % (idx,
                                                                                                  ref_str,
                                                                                                  pred_str)
            

def parse_hallucinations(hal_list):
    lp_dict = defaultdict(list)

    for line in hal_list[1:]:
        lp_idx = line.strip().split('\t')
        assert len(lp_idx) == 4, \
                "Incorrect format, expecting (tab separated): <LP> <SEGMENT_NUMBER> ."

        lp_str = lp_idx[0]
        lp_dict[lp_str.lower()].append(int(lp_idx[1]))

    return lp_dict
                                                                                            
def parse_gold(gold_data, hal_idx):
    
    tag_map = {'GOOD': 1, 'OK': 1, 'BAD': 0, 'good': 1, 'ok': 1, 'bad': 0}
    lp_dict_words = defaultdict(dict)
    lp_dict_tags = defaultdict(dict)
    
    for line in gold_data[1:]:
        gold = line.strip().split('\t')
        assert len(gold) == 6, \
                "Incorrect format, expecting (tab separated): lp, gold, seg_index, target, tokens, tags: "+line
        lp = gold[0]
        curseq = int(gold[2])
        tokens = gold[4]
        tags = gold[5]
        if not curseq in hal_idx[lp]:
            lp_dict_words[lp][curseq]=tokens.split(' ')
            lp_dict_tags[lp][curseq]=[]
            for tag in tags.split(' '):
                lp_dict_tags[lp][curseq].append(tag_map[tag])
        
    return lp_dict_words,lp_dict_tags


def parse_submission( submission_tags_file, hal_idx):
    LP_ID = 0
    SYSNAME = 1
    TYPE = 2
    SENT_ID = 3
    WORD_IDX = 4
    WORD = 5
    TAG = 6

    tag_map = {'GOOD': 1, 'OK': 1, 'BAD': 0, 'good': 1, 'ok': 1, 'bad': 0}

    submission_tags_lines = []
    with codecs.open(submission_tags_file, 'r', encoding='utf-8') as fh:
        submission_tags_lines = fh.readlines()

    disk_footprint = submission_tags_lines[0].rstrip()
    model_params = submission_tags_lines[1].rstrip()
    ensemble = submission_tags_lines[2].rstrip()
    
    lp_dict_words = defaultdict(dict)
    lp_dict_tags = defaultdict(dict)
    for idx, line in enumerate(submission_tags_lines[3:]):
        chunks = line.strip().split('\t')
        lp = chunks[LP_ID]
        
        cur_seq = int(chunks[SENT_ID])
        # submission_words[cur_seq].extend(chunks[WORD].strip().split())
        # submission_tags[cur_seq].append(tag_map[chunks[TAG]])
        if not (lp in lp_dict_words):
            lp_dict_words[lp] = defaultdict(list)
            lp_dict_tags[lp] = defaultdict(list)
            
        if not (cur_seq in hal_idx[lp]):
            # if (lp=='en-mr'):
            #     print(cur_seq)
            #     print(hal_idx[lp])
            lp_dict_words[lp][cur_seq].extend(chunks[WORD].strip().split())
            lp_dict_tags[lp][cur_seq].append(tag_map[chunks[TAG]])
        

    # if not gap_tags:
    #     check_words(ref_words, submission_words)
    #     check_word_tag(submission_words, submission_tags, dataset_name='submission', gap_tags=gap_tags)

    return disk_footprint, model_params, ensemble, lp_dict_words, lp_dict_tags


def flatten(lofl):
    '''
    convert list of lists into a flat list
    '''
    if list_of_lists(lofl):
        return [item for sublist in lofl for item in sublist]
    elif type(lofl) == dict:
        return lofl.values()


def compute_scores(true_tags, test_tags):
    flat_true = flatten(true_tags)
    flat_pred = flatten(test_tags)
   
    f1_all_scores = f1_score(true_tags, test_tags, average=None, pos_label=None)
    # print(f1_all_scores)
    # Matthews correlation coefficient (MCC)
    # true/false positives/negatives
    tp = tn = fp = fn = 0
    for pred_tag, gold_tag in zip(true_tags, test_tags):
        if pred_tag == 1:
            if pred_tag == gold_tag:
                tp += 1
            else:
                fp += 1
        else:
            if pred_tag == gold_tag:
                tn += 1
            else:
                fn += 1

    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = mcc_numerator / mcc_denominator

    return np.append(f1_all_scores, mcc)


# def evaluate(ref_txt_file, ref_tags_file, submission, gap_tags=False):
#     lp_str, disk_footprint, model_params, true_tags, test_tags = parse_submission(ref_txt_file, ref_tags_file, submission, gap_tags)
#     f1_bad, f1_good, mcc = compute_scores(true_tags, test_tags)
    
#     return lp_str, disk_footprint, model_params, f1_bad, f1_good, mcc


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--system', help='System file')
    # parser.add_argument('--gold', help='Gold output file')
    # parser.add_argument('--hallucinations', help='Hallucination index file')
    # parser.add_argument('--baseline', help='Baseline predictions file')
    # parser.add_argument('--outfile', help='Scores.txt  file')
    # args = parser.parse_args()
    
    [_, input_dir, output_dir] = sys.argv
    reference_dir = os.path.join(input_dir, 'ref')
    submission_dir = os.path.join(input_dir, 'res')

    goldlabels_file_name= 'gold_labels_task1_word.tsv'
    baseline_file_name= 'baseline_word_predictions.txt'
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

    print("Loading h indices...")
    with codecs.open(hallucinations_path, 'r', encoding='utf-8') as hal_file:
        hal_idx = parse_hallucinations(hal_file.readlines())
    print("done.")
    
    
    disk_footprint, model_params, ensemble, lp_words_tg, lp_tags_tg = parse_submission(submission_path, hal_idx)
    if compute_baseline:
        _,_,_, lp_words_base, lp_tags_base = parse_submission(baseline_path, hal_idx)

    # with codecs.open(args.gold, 'r', encoding='utf-8') as truth_file:
    with codecs.open(goldlabels_path, 'r', encoding='utf-8') as truth_file:
        lp_words_gold, lp_tags_gold = parse_gold(truth_file.readlines(),hal_idx)
        
    final_score_lines = []
    all_scores = []
    baseline_dict = {}
    
    if compute_baseline:
        for lp in lp_words_base.keys():
            print(lp+'.....')
            lp_gold_words = lp_words_gold[lp]
            lp_gold_tags = lp_tags_gold[lp]
            
            lp_base_words = lp_words_base[lp]
            lp_base_tags = lp_tags_base[lp]
            
            assert len(lp_gold_words) == len(lp_base_words) == len(lp_base_tags) == len(lp_gold_tags),  \
                    "len lp_gold_words: {}, len lp_base_words: {}, len lp_base_tags: {}, len lp_gold_tags: {}.".format(len(lp_gold_words), len(lp_base_words), len(lp_base_tags), len(lp_gold_tags))
            lp_true_tags = []
            lp_baseline_tags = []
            for segment in lp_base_words.keys():
                base_words = lp_base_words[segment]
                base_tags = lp_base_tags[segment]
                gold_words = lp_gold_words[segment]
                gold_tags = lp_gold_tags[segment]
                assert len(base_words) == len(base_tags) == len(gold_tags) == len(gold_words), \
                    "len base_words: {}, len base_tags: {}, len gold_tags: {}, len gold_words: {}\n{}.".format(len(base_words), len(base_tags), len(gold_tags), len(gold_words), segment)
                for wg,wp in zip(base_words,gold_words):
                    assert wg.strip().lower()==wp.strip().lower()
                lp_true_tags.extend(gold_tags)
                lp_baseline_tags.extend(base_tags)
            
            # print(lp_true_tags)
            assert(len(lp_true_tags)==len(lp_baseline_tags))
            f1_bad_base, f1_good_base, mcc_base = compute_scores(lp_true_tags,lp_baseline_tags)
            f1_multi_base = f1_bad_base * f1_good_base
            baseline_dict[lp+'_mcc'] = mcc_base
        baseline_dict[lp+'_f1'] = f1_multi_base
    
    for lp in lp_words_tg.keys():
        print("{}...".format(lp.upper()))
        print("{}...".format(lp.upper()), file=sys.stderr)
        lp_gold_words = lp_words_gold[lp]
        lp_gold_tags = lp_tags_gold[lp]
        
        lp_pred_words = lp_words_tg[lp]
        lp_pred_tags = lp_tags_tg[lp]
        
        assert len(lp_gold_words) == len(lp_pred_words) == len(lp_pred_tags) == len(lp_gold_tags)
        lp_true_tags = []
        lp_predicted_tags = []
        for segment in lp_pred_words.keys():
            pred_words = lp_pred_words[segment]
            pred_tags = lp_pred_tags[segment]
            gold_words = lp_gold_words[segment]
            gold_tags = lp_gold_tags[segment]
            assert len(pred_words) == len(pred_tags) == len(gold_tags) == len(gold_words)
            for wg,wp in zip(pred_words,gold_words):
                assert wg.strip().lower()==wp.strip().lower()
            lp_true_tags.extend(gold_tags)
            lp_predicted_tags.extend(pred_tags)
        
        f1_bad_tg, f1_good_tg, mcc_tg = compute_scores(lp_true_tags,lp_predicted_tags)
        f1_multi_tg = f1_bad_tg * f1_good_tg
        all_scores.append([mcc_tg, f1_multi_tg])

        final_score_lines.append(lp.replace('-','')+"_mcc: {:.4}".format(mcc_tg))
        print(lp.replace('-','')+"_mcc: {:.4}".format(mcc_tg))
        print(lp.replace('-','')+"_mcc: {:.4}".format(mcc_tg), file=sys.stderr)
        final_score_lines.append(lp.replace('-','')+"_f1: {:.4}".format(f1_multi_tg))
        print(lp.replace('-','')+"_f1: {:.4}".format(f1_multi_tg))
        print(lp.replace('-','')+"_f1: {:.4}".format(f1_multi_tg), file=sys.stderr)
        # final_score_lines.append('\n')
        
        if compute_baseline:
            if mcc_tg >= baseline_dict[lp+'_mcc']:
                print('Above baseline for MCC (primary metric) by: {:.4}'.format(mcc_tg - baseline_dict[lp+'_mcc']))
                print('Above baseline for MCC (primary metric) by: {:.4}'.format(mcc_tg - baseline_dict[lp+'_mcc']), file=sys.stderr)
            else:
                print('Below baseline for MCC (primary metric) by: {:.4}'.format(mcc_tg - baseline_dict[lp+'_mcc']))
                print('Below baseline for MCC (primary metric) by: {:.4}'.format(mcc_tg - baseline_dict[lp+'_mcc']), file=sys.stderr)
        print("----")
        print("----", file=sys.stderr)


    print("\t averaging...")
    average_scores = np.mean(np.array(all_scores), axis=0)
    print("done.")
    # final_score_lines.append('\n---------- MULTILINGUAL PERFORMANCE ------------\n')
    if set(lp_words_gold.keys())==set(lp_words_tg.keys()):
        
        final_score_lines.append("avg_mcc: {:.4}".format(average_scores[0]))
        final_score_lines.append("avg_f1: {:.4}".format(average_scores[1]))
        print("avg_mcc: {:.4}".format(average_scores[0]))
        print("avg_mcc: {:.4}".format(average_scores[0]), file=sys.stderr)
        print("avg_f1: {:.4}".format(average_scores[1]))
        print("avg_f1: {:.4}".format(average_scores[1]), file=sys.stderr)
    else:
        final_score_lines.append("avg_mcc: n/a")
        final_score_lines.append("avg_f1: n/a")
        print("avg_mcc: n/a")
        print("avg_mcc: n/a", file=sys.stderr)
        print("avg_f1: n/a")
        print("avg_f1: n/a", file=sys.stderr)
    
    # final_score_lines.append('\n------------------------------------------------\n')
    final_score_lines.append("ensembles: {}".format(ensemble))
    final_score_lines.append("model_params: {}".format(model_params))
    final_score_lines.append("disk footprint: {}".format(disk_footprint))
    print("ensembles: {}".format(ensemble))
    print("ensembles: {}".format(ensemble), file=sys.stderr)
    print("model_params: {}".format(model_params))
    print("model_params: {}".format(model_params), file=sys.stderr)
    print("disk footprint: {}".format(disk_footprint))
    print("disk footprint: {}".format(disk_footprint), file=sys.stderr)
    
    # with open(args.outfile,'w') as wf:
    with open(os.path.join(output_dir, 'scores.txt'), 'w', encoding='utf-8') as wf:
        for line in final_score_lines:
            wf.write(line+'\n')


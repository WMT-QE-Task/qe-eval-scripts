"""Script to evaluate document-level QE as in the WMT19 shared task."""

import argparse
from collections import defaultdict
import numpy as np
import codecs
import os
import sys


def load_annotations(file_path,hal_idx,isgold):
    """Loads a file containing annotations for multiple documents.

    The file should contain lines with the following format:
    <LANGUAGE PAIR> <METHOD NAME> <SEGMENT NUMBER> <TARGET SENTENCE> <ERROR START INDICES> <ERROR END INDICES> <ERROR TYPES>

    Fields are separated by tabs; LINE, SPAN START POSITIONS and SPAN LENGTHS
    can have a list of values separated by white space.

    Args:
        file_path: path to the file.
    Returns:
        a dictionary mapping document id's to a list of annotations.
    """
    with open(file_path, 'r', encoding='utf8') as f:
        lpdict = {}
        for i, line in enumerate(f):
            if i==0 and (not isgold):
                params = line.strip()
            elif i==1 and (not isgold):
                size = line.strip()
            elif i==2 and (not isgold):
                ensemble = int(line.strip())
            elif i>=3 or (isgold and i>=1):
                line = line.strip()
                if not line:
                    continue
                fields = line.split('\t')
                assert len(fields)==7
                
                lp = fields[0]
                
                mt = fields[3].strip()
                start = fields[4]
                end = fields[5]
                error = fields[6]
                if not lp in lpdict:
                    lpdict[lp]={}
                      
                seg_id = fields[2]
                # if not  (seg_id in hal_idx[lp]):
                if not  (int(seg_id) in hal_idx[lp]):
                    
                    lpdict[lp][seg_id]={} 
                    lpdict[lp][seg_id]['major']=[0]*(len(mt)+1)
                    lpdict[lp][seg_id]['minor']=[0]*(len(mt)+1)
                    for s,e,t in zip(start.split(' '),end.split(' '),error.split(' ')):
                        s = int(s)
                        e = int(e)
                        if e>len(mt):
                            e=len(mt)
                        
                        if s!=-1 and e!=-1 :
                            if s==e:
                                lpdict[lp][seg_id][t][s]+=1
                            else:
                                i=s
                                while i<e:
                                    lpdict[lp][seg_id][t][i]+=1
                                    i+=1 
    if isgold:
        return lpdict
    else:
                   
        return params,size,ensemble,lpdict


def parse_hallucinations(hal_list):
    lp_dict = defaultdict(list)

    for line in hal_list[1:]:
        lp_idx = line.strip().split('\t')
        assert len(lp_idx) == 2, \
                "Incorrect format, expecting (tab separated): <LP> <SEGMENT_NUMBER> ."

        lp_str = lp_idx[0]
        lp_dict[lp_str.lower()].append(int(lp_idx[1]))

    return lp_dict

def score_new(system,reference):
    lp_results = {}
    for lp in system.keys():
        assert lp in reference.keys(), lp+ " is not in the gold data language pairs"
        system_preds = system[lp]
        gold_labels = reference[lp]
        recall=0
        precision=0
        f1=0
        
        tp=0
        tn=0
        fp=0
        fn=0
        total_sys = 0 
        total_gold = 0  
        for segid in gold_labels:
            assert len(gold_labels[segid]['major']) == len(system_preds[segid]['major']), str(len(gold_labels[segid]['major']))+"-"+str( len(system_preds[segid]['major']))+'-'+str(segid)+str(lp)
            
            for character_gold_major, character_sys_major, character_gold_minor, character_sys_minor in zip(gold_labels[segid]['major'],system_preds[segid]['major'],gold_labels[segid]['minor'],system_preds[segid]['minor']):
                if character_gold_major!=0 or character_gold_minor!=0:
                    total_gold+=1
                if character_sys_major!=0 or character_sys_minor!=0:
                    total_sys+=1
                if character_gold_major==0 and character_gold_minor==0 :
                    if character_sys_major==0 and character_sys_minor==0:
                        tn+=1
                    else:
                        
                        fp+=1 
                else:
                    if character_gold_major>0 and character_gold_minor==0 :
                        if character_sys_major>0  :
                            tp+=1 
                        elif character_sys_minor>0:
                            tp+=0.5
                    elif character_gold_minor>0 and character_gold_major==0 :
                        if character_sys_minor>0  :
                            tp+=1 
                        elif character_sys_major>0:
                            tp+=0.5
                    elif character_gold_minor>0 and character_gold_major>0 :
                        if character_sys_minor>0 or character_sys_major>0:
                            tp+=1
                    
    
                    
        seg_recall1 = tp/(total_gold)
        seg_precision1 = tp/(total_sys)
        f11 = 2 * seg_precision1 * seg_recall1 / (seg_precision1 + seg_recall1)
        
        lp_results[lp]=[seg_recall1,seg_precision1,f11] 
    return lp_results

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--system', help='System file')
    # parser.add_argument('--gold', help='Gold output file')
    # parser.add_argument('--hallucinations', help='Hallucination index file')
    # parser.add_argument('--baseline', help='Baseline predictions file')
    # parser.add_argument('--outfile', help='Scores.txt  file')
    #
    # args = parser.parse_args()

    [_, input_dir, output_dir] = sys.argv
    reference_dir = os.path.join(input_dir, 'ref')
    submission_dir = os.path.join(input_dir, 'res')

    goldlabels_file_name= 'gold_labels_task2_error_span.tsv'
    baseline_file_name= 'baseline_predictions_task2.tsv'
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

    # print("Loading h indices...")
    # with codecs.open(args.hallucinations, 'r', encoding='utf-8') as hal_file:
    with codecs.open(hallucinations_path, 'r', encoding='utf-8') as hal_file:
        hal_idx = parse_hallucinations(hal_file.readlines())
    # print("done.")
    
    # params,size,ensemble,system = load_annotations(args.system, hal_idx,False)
    params,size,ensemble,system = load_annotations(submission_path, hal_idx,False)

    
    reference = load_annotations(goldlabels_path,hal_idx,True)
      
    if compute_baseline:
        _,_,_,baseline = load_annotations(baseline_path, hal_idx,False)
        lpbase = score_new(baseline,reference)
 
    lpsys = score_new(system,reference)
    
    all_scores = []
    for lp in lpsys:
        all_scores.append(lpsys[lp])
      
    final_score_lines = []
    # print("\t averaging...")
    average_scores = np.mean(np.array(all_scores), axis=0)
    # print("done.")    
    
    for lp in lpsys:
        print(lp.upper()+":")
        print(lp.upper()+":", file=sys.stderr)
        final_score_lines.append(lp.replace('-','')+"_f1: {:.4}".format(lpsys[lp][2]))
        print(lp.replace('-','')+"_f1: {:.4}".format(lpsys[lp][2]))
        print(lp.replace('-','')+"_f1: {:.4}".format(lpsys[lp][2]), file=sys.stderr)
        final_score_lines.append(lp.replace('-','')+"_rec: {:.4}".format(lpsys[lp][0]))
        print(lp.replace('-','')+"_rec: {:.4}".format(lpsys[lp][0]))
        print(lp.replace('-','')+"_rec: {:.4}".format(lpsys[lp][0]), file=sys.stderr)
        final_score_lines.append(lp.replace('-','')+"_prec: {:.4}".format(lpsys[lp][1]))
        print(lp.replace('-','')+"_prec: {:.4}".format(lpsys[lp][1]))
        print(lp.replace('-','')+"_prec: {:.4}".format(lpsys[lp][1]), file=sys.stderr)
        if compute_baseline:
            if lpsys[lp][2] >= lpbase[lp][2]:
                print('Above baseline for F1 (primary metric) by: {:.4}'.format(lpsys[lp][2] - lpbase[lp][2]))
                print('Above baseline for F1 (primary metric) by: {:.4}'.format(lpsys[lp][2] - lpbase[lp][2]), file=sys.stderr)
            else:
                print('Below baseline for F1 (primary metric) by: {:.4}'.format(lpsys[lp][2] - lpbase[lp][2]))
                print('Below baseline for F1 (primary metric) by: {:.4}'.format(lpsys[lp][2] - lpbase[lp][2]), file=sys.stderr)
                
        print("-----")
        print("-----", file=sys.stderr)
    
   
    average_scores = np.mean(np.array(all_scores), axis=0)

    
    # final_score_lines.append('\n---------- MULTILINGUAL PERFORMANCE ------------\n')
    if set(system.keys())==set(reference.keys()):
        final_score_lines.append("avg_f1: {:.4}".format(average_scores[2]))
        final_score_lines.append("avg_rec: {:.4}".format(average_scores[0]))
        final_score_lines.append("avg_prec: {:.4}".format(average_scores[1]))
        print("avg_f1: {:.4}".format(average_scores[2]))
        print("avg_f1: {:.4}".format(average_scores[2]), file=sys.stderr)
        print("avg_rec: {:.4}".format(average_scores[0]))
        print("avg_rec: {:.4}".format(average_scores[0]), file=sys.stderr)
        print("avg_prec: {:.4}".format(average_scores[1]))
        print("avg_prec: {:.4}".format(average_scores[1]), file=sys.stderr)
    else:
        final_score_lines.append("avg_f1: n/a")
        final_score_lines.append("avg_prec: n/a")
        final_score_lines.append("avg_rec: n/a")
        print("avg_f1: n/a")
        print("avg_f1: n/a", file=sys.stderr)
        print("avg_rec: n/a")
        print("avg_rec: n/a", file=sys.stderr)
        print("avg_prec: n/a")
        print("avg_prec: n/a", file=sys.stderr)
    
    
    # final_score_lines.append('\n------------------------------------------------\n')
    final_score_lines.append("ensembles: {}".format(ensemble))
    final_score_lines.append("model_params: {}".format(params))
    final_score_lines.append("disk footprint: {}".format(size))
    print("ensembles: {}".format(ensemble))
    print("ensembles: {}".format(ensemble), file=sys.stderr)
    print("model_params: {}".format(params))
    print("model_params: {}".format(params), file=sys.stderr)
    print("disk footprint: {}".format(size))
    print("disk footprint: {}".format(size), file=sys.stderr)
    
    # with open(args.outfile,'w') as wf:
    with open(os.path.join(output_dir, 'scores.txt'), 'w', encoding='utf-8') as wf:
        for line in final_score_lines:
            wf.write(line+'\n')

if __name__ == '__main__':
    main()

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from argparse import ArgumentParser
import json
import copy
from tqdm import tqdm

from label2_generator_3steps_utt import label2_multi_generate
from dstc12.utils import  DotAllRegexParser
from proc_fuc import preproce_dataset


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('org_dataset_file', type=str)
    parser.add_argument('clusters_with_preferences', type=str)
    parser.add_argument('result_file', type=str)
    parser.add_argument('--n-clusters', type=int, default=10)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--llm-name', type=str)
    return parser.parse_args()


def get_theme(dataset):
    themed_utts = list()
    for conv in dataset:
        for utt in conv:
            if utt['theme_label'] != None:
                themed_utts.append(utt[utt['role']])
    return themed_utts


def extract_dataset(seg_dataset, tokenizer = None, only_themed = True):
    if tokenizer == None:
        eot = ""
        start_h = ""
        end_h = ""
    else:
        eot = '<|eot_id|>'
        start_h = '<|start_header_id|>'
        end_h = '<|end_header_id|>'

    if only_themed:
        themed_top_dataset = list()
        top_ids = list()
        for conv in tqdm(seg_dataset):
            for utt in conv:
                if utt['theme_label'] != None:
                    target_top = utt['start_top']
                    top_seq = ""
                    target_id = utt['utterance_id']
                    # exctract the same utt from the same topic and make them a sequence.
                    for utt in conv:
                        if utt['start_top'] == target_top:
                            top_seq = top_seq + start_h + utt['role'] + ':' + end_h + utt[utt['role']] + eot
                        elif utt['start_top'] > target_top:
                            break
                    themed_top_dataset.append(top_seq)
                    top_ids.append(target_id)
        return top_ids, themed_top_dataset

    else:
        top_dataset = list()
        top_ids = list()
        for conv in tqdm(seg_dataset):
            for utt in conv:
                target_top = utt['start_top']
                top_seq = ""
                target_id = utt['utterance_id']
                # exctract the same utt from the same topic and make them a sequence.
                for utt in conv:
                    if utt['start_top'] == target_top:
                        top_seq = top_seq + start_h + utt['role'] + ':' + end_h + utt[utt['role']] + eot
                    elif utt['start_top'] > target_top:
                        break
                top_dataset.append(top_seq)
                top_ids.append(target_id)
        return top_ids, top_dataset



def main(dataset, clusters_with_preferences, llm_name, n_clusters):
    
    utterances = get_theme(dataset)
    clustered_utterances = [[] for _ in range(n_clusters)]
    for i, label in enumerate(clusters_with_preferences):
        clustered_utterances[label].append(utterances[i])
    cluster_label_map = {}
    cluster_label_map = label2_multi_generate(llm_name, n_clusters, clusters_with_preferences, utterances, utterances)
    return cluster_label_map



if __name__ == '__main__':
    args = parse_args()

    with open(args.org_dataset_file) as f:
        org_dataset = [json.loads(line) for line in f]
    dataset = preproce_dataset(org_dataset) 
        
    with open(args.clusters_with_preferences) as f:
        clusters_with_preferences = json.load(f)
        
        
    cluster_label_map = main(
        dataset,
        clusters_with_preferences,
        args.llm_name,
        args.n_clusters,
    )

    dataset_predicted = copy.deepcopy(org_dataset)

    print(len( cluster_label_map))
    for dialogue in dataset_predicted:
        for turn in dialogue['turns']:
            if turn['theme_label'] is not None:
                try:
                    turn['theme_label_predicted'] = cluster_label_map[turn['utterance']]
                except:
                    turn['theme_label_predicted'] = "your theme label"
                
    with open(args.result_file, 'w') as result_out:
        for dialogue in dataset_predicted:
            print(json.dumps(dialogue), file=result_out)

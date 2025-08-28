# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from argparse import ArgumentParser
import json
import os
import copy
import collections
import torch
import numpy as np

import getpass
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances
import umap

from datasets import Dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, BitsAndBytesConfig
from label2_generator_3steps_utt import label2_base_generate, label2_multi_generate, gpt4_multi_generate
from dstc12.prompts import LABEL_CLUSTERS_PROMPT
from dstc12.utils import get_llm, DotAllRegexParser
from proc_fuc import seg_create, add_seg, seg_create_new, add_seg_new


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('org_dataset_file', type=str)
    parser.add_argument('dataset_file', type=str)
    parser.add_argument('preferences_file', type=str)
    parser.add_argument('slice_file', type=str)
    parser.add_argument('score_matrix', type=str)
    parser.add_argument('result_file', type=str)
    parser.add_argument('--n-clusters', type=int, default=10)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--embedding-model-name', type=str, default='/share/home/kerui/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0')
    parser.add_argument('--llm-name', type=str, default='/share/102/shared_models/01_LLM-models/Meta-Llama-3-8B-Instruct')
    return parser.parse_args()


def find_second_closest_cluster(emb, centroids):
    distances = [np.linalg.norm(emb - centroid) for centroid in centroids]
    sorted_indices = np.argsort(distances)
    return sorted_indices[1]


def get_weighted_distance(ord_dist_matrix, score_matrix):
    inv_matrix = (0 - score_matrix)
    abs_matrix = inv_matrix + abs(inv_matrix.min()) + 1
    # scalar the data into 1-2
    adj_matrix = ((abs_matrix - torch.min(abs_matrix)) / (torch.max(abs_matrix) - torch.min(abs_matrix))) + 1
    adj_matrix = adj_matrix.squeeze(0)

    len_top = len(ord_dist_matrix)
    num_ls = [i for i in range(1,len_top+1)]
    ind_ls = np.cumsum(num_ls)

    # initializer Col_list and make it as [[a11]]
    col_list = [adj_matrix[0:1]]
    # then add other col and make it become  [[a11],[a12, a22],[a13,a23,a33],....]
    for i in range(1,len(ind_ls)):
        col = adj_matrix[ind_ls[i-1]:ind_ls[i]]
        col_list.append(col)
    col_list
    mut_matrix = torch.nn.utils.rnn.pad_sequence(col_list)
    # make the mut_matrix (upper_triangular_matrix) become symmetric matrix
    weight_matrix = mut_matrix.triu() + mut_matrix.triu(1).T
    weight_dist_matrix = weight_matrix.to("cuda:0") * torch.Tensor(ord_dist_matrix).to("cuda:0")
    
    return weight_dist_matrix


def arrange_new_cluster(weighted_distance, cluster_labels, topic_idx_mapping, topic_cluster_mapping, top_id_a, top_id_b, type):
    if type == 'cannot_link':
        modified_cluster_labels = copy.deepcopy(cluster_labels)
        a_idx = topic_idx_mapping[top_id_a]
        b_idx = topic_idx_mapping[top_id_b]
        co_label = topic_cluster_mapping[top_id_a]
        label_list = list(set(cluster_labels))
        
        # for a and b, whose average dist to co_label's points is larger, will be arranged to another cluster.
        co_idxs = [index for index, label in enumerate(cluster_labels) if label == co_label]
        a_dist_to_co = list()
        b_dist_to_co = list()
        for co_idx in co_idxs:
            a_dist_to_co.append(weighted_distance[a_idx, co_idx])
            b_dist_to_co.append(weighted_distance[b_idx, co_idx])
        # rearrange a
        if sum(a_dist_to_co) > sum(b_dist_to_co):
            a_sm_dists = list()
            a_sm_labels = list()
            for sm_label in label_list:
                if sm_label != co_label:
                    sm_idxs = [index for index, label in enumerate(cluster_labels) if sm_label == label]
                    a_dist_to_sm = list()
                    for sm_idx in sm_idxs:
                        a_dist_to_sm.append(weighted_distance[a_idx, sm_idx])
                    a_sm_dists.append(sum(a_dist_to_sm))
                    a_sm_labels.append(sm_label)
                    
            target_dist = max(a_sm_dists)
            target_label_idx = 0
            for i, dist in enumerate(a_sm_dists):
                if dist <= target_dist:
                    target_label_idx = i
            a_new_label = a_sm_labels[target_label_idx]
            modified_cluster_labels[a_idx] = a_new_label
            
        # rearrange b
        if sum(a_dist_to_co) <= sum(b_dist_to_co):
            b_sm_dists = list()
            b_sm_labels = list()
            for sm_label in label_list:
                if sm_label != co_label:
                    sm_idxs = [index for index, label in enumerate(cluster_labels) if sm_label == label]
                    b_dist_to_sm = list()
                    for sm_idx in sm_idxs:
                        b_dist_to_sm.append(weighted_distance[b_idx, sm_idx])
                    b_sm_dists.append(sum(b_dist_to_sm))
                    b_sm_labels.append(sm_label)
                    
            target_dist = max(b_sm_dists)
            target_label_idx = 0
            for i, dist in enumerate(b_sm_dists):
                if dist <= target_dist:
                    target_label_idx = i
            b_new_label = b_sm_labels[target_label_idx]
            modified_cluster_labels[b_idx] = b_new_label
        return modified_cluster_labels
    
    
    if type == "should_link":
        # for a and b, they will be arranged to the same cluster (choose from a's or b's cluster) in which their sum distance is the smallest .
        modified_cluster_labels = copy.deepcopy(cluster_labels)
        a_idx = topic_idx_mapping[top_id_a]
        b_idx = topic_idx_mapping[top_id_b]
        a_label = topic_cluster_mapping[top_id_a]
        b_label = topic_cluster_mapping[top_id_b]
        label_list = list(set(cluster_labels))
        
        # if cluster both a and b to a_label
        sm_a_idxs = [index for index, label in enumerate(cluster_labels) if label == a_label]
        b_dist_to_sm = list()
        a_dist_to_sm = list()
        for idx in sm_a_idxs:
            b_dist_to_sm.append(weighted_distance[b_idx, idx])
            a_dist_to_sm.append(weighted_distance[a_idx, idx])
        total_a_dist = sum(b_dist_to_sm) + sum(a_dist_to_sm)
        
        # if cluster both a and b to b_label
        sm_b_idxs = [index for index, label in enumerate(cluster_labels) if label == b_label]
        b_dist_to_sm = list()
        a_dist_to_sm = list()
        for idx in sm_b_idxs:
            b_dist_to_sm.append(weighted_distance[b_idx, idx])
            a_dist_to_sm.append(weighted_distance[a_idx, idx])
        total_b_dist = sum(b_dist_to_sm) + sum(a_dist_to_sm)
        
        if total_a_dist > total_b_dist:
            modified_cluster_labels[a_idx] = b_label
        else:
            modified_cluster_labels[b_idx] = a_label
        return modified_cluster_labels
            
        
        

def apply_preferences_to_clusters(topics, top_ids, weighted_distance, cluster_labels, shouldlink_pairs, cannot_link_pairs):
    assert len(topics) == len(cluster_labels)
    topic_cluster_mapping = collections.defaultdict(lambda: -1)
    topic_idx_mapping = collections.defaultdict(lambda: -1)
    
    for idx, cluster_label in enumerate(cluster_labels):
        top_id = top_ids[idx]
        topic_cluster_mapping[top_id] = cluster_label
        topic_idx_mapping[top_id] = idx
    modified_cluster_labels = copy.deepcopy(cluster_labels)
    print(f"should_link processing...")
    num_modified = 0
    for top_id_a,  top_id_b in tqdm(shouldlink_pairs):
        cluster_a, cluster_b = topic_cluster_mapping[top_id_a], topic_cluster_mapping[top_id_b]
        if cluster_a != cluster_b:
            num_modified = num_modified + 1
            modified_cluster_labels = arrange_new_cluster(weighted_distance, modified_cluster_labels, topic_idx_mapping, topic_cluster_mapping, top_id_a, top_id_b, type='should_link')
    print(f"number of modified pair: {num_modified}")
    print(f"cannot_link processing...")
    num_modified = 0
    for top_id_a, top_id_b in tqdm(cannot_link_pairs):
        cluster_a, cluster_b = topic_cluster_mapping[top_id_a], topic_cluster_mapping[top_id_b]
        if cluster_a == cluster_b:
            num_modified = num_modified + 1
            modified_cluster_labels = arrange_new_cluster(weighted_distance, modified_cluster_labels, topic_idx_mapping, topic_cluster_mapping, top_id_a, top_id_b, type='cannot_link')
    print(f"number of modified pair: {num_modified}")
    return modified_cluster_labels
    
    # datapoint_modification_counter = collections.defaultdict(lambda: 0)

    # utterance_cluster_mapping = collections.defaultdict(lambda: -1)
    # utterance_idx_mapping = collections.defaultdict(lambda: -1)
    # for utt_idx, cluster_label in enumerate(cluster_labels):
    #     utterance = topics[utt_idx]
    #     utterance_cluster_mapping[utterance] = cluster_label
    #     utterance_idx_mapping[utterance] = utt_idx
    # modified_cluster_labels = copy.deepcopy(cluster_labels)
    # for utt_a, utt_b in shouldlink_pairs:
    #     cluster_a, cluster_b = utterance_cluster_mapping[utt_a], utterance_cluster_mapping[utt_b]
    #     if cluster_a != cluster_b:
    #         utt_b_idx = utterance_idx_mapping[utt_b]
    #         modified_cluster_labels[utt_b_idx] = cluster_a
    #         utterance_cluster_mapping[utt_b] = cluster_a
    #         datapoint_modification_counter[utt_b_idx] += 1
    # for utt_a, utt_b in cannot_link_pairs:
    #     cluster_a, cluster_b = utterance_cluster_mapping[utt_a], utterance_cluster_mapping[utt_b]
    #     if cluster_a == cluster_b:
    #         utt_b_idx = utterance_idx_mapping[utt_b]
    #         utt_b_new_cluster = find_second_closest_cluster(topic_embs[utt_b_idx], cluster_centroids)
    #         modified_cluster_labels[utt_b_idx] = utt_b_new_cluster
    #         utterance_cluster_mapping[utt_b] = utt_b_new_cluster
    #         datapoint_modification_counter[utt_b_idx] += 1
    # return modified_cluster_labels

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

def tokenize(example):
    data = example['conversations']
    tokenized_data = tokenizer(data, return_tensors="pt",truncation=True, max_length=2048, padding=True)
    return {"input_ids" : tokenized_data['input_ids'].squeeze(0),
            "attention_mask": tokenized_data['attention_mask'].squeeze(0)
    }

def get_embedding(themed_top_dataset, tokenizer, model):
    topics_dataset = Dataset.from_dict({'conversations': themed_top_dataset})
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    # model.to('cuda:0')
    # model = torch.nn.DataParallel(model, device_ids=[0,1,2])
    tokenizer = tokenizer
    embedding_model_name = embedding_model_name
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model.eval()
    tokenized_dataset = topics_dataset.map(tokenize, remove_columns = topics_dataset.column_names)
    
    batch_size = 10
    dataloader = DataLoader(
        dataset = tokenized_dataset,
        batch_size = batch_size,
        collate_fn= DataCollatorWithPadding(tokenizer, padding="max_length", max_length=2048, return_tensors="pt")
    )
    
    query_embeddings = list()
    for batch in tqdm(dataloader):
        batch = {k:v.to('cuda:0') for k, v in batch.items()}
        with torch.no_grad():
            embeddings = model(**batch)
            query_embeddings = query_embeddings + embeddings.tolist()
    print(query_embeddings)
    print(len(query_embeddings))
    dist_matrix = euclidean_distances(query_embeddings)
    return dist_matrix


def main(dataset, slice_data, score_matrix, linking_preferences, embedding_model_name, llm_name, n_clusters, random_state):
    llm = get_llm(llm_name)
    
    utterances = get_theme(dataset)
    bound_ls = seg_create_new(slice_data)
    seg_dataset = add_seg_new(dataset, bound_ls)
    top_ids, topics = extract_dataset(seg_dataset)
    
    # accelerator = Accelerator()
    embedder = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": 'cuda'})

    # topic level embedding
    topic_embeddings = [embedder.embed_query(top) for top in tqdm(topics)]
    # utterance level embedding
    query_embeddings = [embedder.embed_query(utt) for utt in tqdm(utterances)]

    
    # Dimension Reduction
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    query_embeddings = reducer.fit_transform(query_embeddings)
    
    
    # Use spectral
    spectral = SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans', random_state=42)
    spectral.fit(query_embeddings)
    clusters = spectral.labels_
    
    # use  kmeans
    # kmeans = KMeans(n_clusters=n_clusters, n_init=1, init='k-means++', random_state=random_state)
    # kmeans.fit(query_embeddings)
    # clusters = kmeans.labels_
    # # centroids = kmeans.cluster_centers_
    
    ord_dist_matrix = euclidean_distances(topic_embeddings )
    weighted_distance = get_weighted_distance(ord_dist_matrix, score_matrix)
    
    clusters_with_preferences = apply_preferences_to_clusters(
        utterances,
        top_ids,
        weighted_distance,
        clusters,
        linking_preferences['should_link'],
        linking_preferences['cannot_link']
    )
    return clusters_with_preferences



if __name__ == '__main__':
    args = parse_args()

    # if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        # os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")
        
    with open(args.org_dataset_file) as f:
        org_dataset = [json.loads(line) for line in f]
    dataset = preproce_dataset(org_dataset)
        
    with open(args.slice_file) as f:
        slice_data = [json.loads(line) for line in f]
    with open(args.preferences_file) as prefs_in:
        linking_preferences = json.load(prefs_in)
    score_matrix = torch.load(args.score_matrix)
        
    cluster_label_map = main(
        dataset,
        slice_data,
        score_matrix,
        linking_preferences,
        args.embedding_model_name,
        args.llm_name,
        args.n_clusters,
        args.random_state
    )
                
    with open(args.result_file, 'w') as fw:
        json.dumps(cluster_label_map, fw)

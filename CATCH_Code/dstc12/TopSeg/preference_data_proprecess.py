import os
import re
import json
import torch
import random
import pickle
import IPython
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from collections import defaultdict, Counter
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModel, set_seed, BertForNextSentencePrediction
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
set_seed(3407)

def load_dialogue_pairs():
    id2utt = {}
    with open(args.dataset, 'r') as lines:
        for line in lines:
            line = json.loads(line)
            for turn in line['turns']:
                id2utt[turn['utterance_id']] = turn['utterance']
    
    positive_pairs = []
    negative_pairs = []
    with open(args.preference_data, 'r') as f:
        pairs = json.load(f)
        for pos_pairs in pairs['should_link']:
            positive_pairs.append([id2utt[pos_pairs[0]], id2utt[pos_pairs[1]]])
        for neg_pairs in pairs['cannot_link']:
            negative_pairs.append([id2utt[neg_pairs[0]], id2utt[neg_pairs[1]]])
    
    return positive_pairs, negative_pairs

def process_dialogue_pairs(positive_pairs, negative_pairs, args):

    data = []
    topic_data = []
    
    # clean all positive samples
    cleaned_positive_pairs = []
    for p_a, p_b in positive_pairs:
        p_a_clean = re.sub(r'\s([,?.!"](?:\s|$))', r'\1', p_a)
        p_b_clean = re.sub(r'\s([,?.!"](?:\s|$))', r'\1', p_b)
        cleaned_positive_pairs.append((p_a_clean, p_b_clean))
    
    # clean all negative samples 
    cleaned_negative_pairs = []
    for n_a, n_b in negative_pairs:
        n_a_clean = re.sub(r'\s([,?.!"](?:\s|$))', r'\1', n_a)
        n_b_clean = re.sub(r'\s([,?.!"](?:\s|$))', r'\1', n_b)
        cleaned_negative_pairs.append((n_a_clean, n_b_clean))
    
    for i, (p_a_clean, p_b_clean) in enumerate(tqdm(cleaned_positive_pairs, desc="Processing positive-negative pairs")):
        for j, (n_a_clean, n_b_clean) in enumerate(cleaned_negative_pairs):

            # sample format：[(pos_context, pos_response), (neg_context, neg_response)]
            data.append([([p_a_clean], [p_b_clean]), ([n_a_clean], [n_b_clean])])
            
            full_dialogue = [p_a_clean, p_b_clean]
            topic_data.append((full_dialogue, 1)) 
    
    print(f"Generated {len(data)} training samples from {len(cleaned_positive_pairs)} positive and {len(cleaned_negative_pairs)} negative pairs")
    
    return data, topic_data

def process_dialogue_pairs_enhanced(positive_pairs, negative_pairs, args):
    data = []
    topic_data = []
    

    cleaned_positive_pairs = []
    for p_a, p_b in positive_pairs:
        p_a_clean = re.sub(r'\s([,?.!"](?:\s|$))', r'\1', p_a)
        p_b_clean = re.sub(r'\s([,?.!"](?:\s|$))', r'\1', p_b)
        cleaned_positive_pairs.append((p_a_clean, p_b_clean))
    

    cleaned_negative_pairs = []
    for n_a, n_b in negative_pairs:
        n_a_clean = re.sub(r'\s([,?.!"](?:\s|$))', r'\1', n_a)
        n_b_clean = re.sub(r'\s([,?.!"](?:\s|$))', r'\1', n_b)
        cleaned_negative_pairs.append((n_a_clean, n_b_clean))
    
    for i, (p_a_clean, p_b_clean) in enumerate(tqdm(cleaned_positive_pairs, desc="Processing positive-negative pairs")):
        for j, (n_a_clean, n_b_clean) in enumerate(cleaned_negative_pairs):

            data.append([([p_a_clean], [p_b_clean]), ([n_a_clean], [n_b_clean])])
            
            # topic_data
            full_dialogue = [p_a_clean, p_b_clean]
            topic_data.append((full_dialogue, 1))
        

        if args.add_hard_negatives and len(cleaned_positive_pairs) > 1:
            other_positives = [pair for k, pair in enumerate(cleaned_positive_pairs) if k != i]

            num_hard_neg = min(len(other_positives), args.max_hard_negatives if hasattr(args, 'max_hard_negatives') else 2)
            selected_hard_neg = random.sample(other_positives, num_hard_neg)
            
            for hard_p_a, hard_p_b in selected_hard_neg:
                data.append([([p_a_clean], [p_b_clean]), ([hard_p_a], [hard_p_b])])
                
                full_dialogue = [p_a_clean, p_b_clean]
                topic_data.append((full_dialogue, 1))
    
    print(f"Generated {len(data)} training samples from {len(cleaned_positive_pairs)} positive and {len(cleaned_negative_pairs)} negative pairs")
    
    return data, topic_data

def main(args):
    MAX_LEN = 512
    

    positive_pairs, negative_pairs = load_dialogue_pairs()
    

    data, topic_data = process_dialogue_pairs(positive_pairs, negative_pairs, args)
    
    print(f"Processed {len(data)} dialogue pairs")
    

    json.dump(data, open(f'{args.output_dir}/processed_pairs_{args.version}.json', 'w'))
    json.dump(topic_data, open(f'{args.output_dir}/topic_data_{args.version}.json', 'w'))
    

    turn_ids, id_inputs, topic_inputs, sample_num_memory, topic_train, topic_num = [], [], [], [len(i) for i in data], [], []
    
    for i in tqdm(range(len(data)), desc="Tokenizing"):
        for sample in data[i]:
            context, cur = sample
            

            sent1 = context[0] if context else ""
            sent2 = cur[0] if cur else ""
            

            encoded_sent1 = tokenizer.encode(sent1, add_special_tokens=True, return_tensors='pt')
            encoded_sent2 = tokenizer.encode(sent2, truncation=True, max_length=256, add_special_tokens=True, return_tensors='pt')
            

            topic_con = tokenizer(context, truncation=True, max_length=256)
            topic_cur = tokenizer(cur, truncation=True, max_length=256)
            

            id_input = encoded_sent1[0].tolist()[-257:] + encoded_sent2[0].tolist()[1:]
            turn_id = [0] * len(encoded_sent1[0].tolist()[-257:]) + [1] * len(encoded_sent2[0].tolist()[1:])
            
            id_inputs.append(torch.Tensor(id_input))
            topic_inputs.append((topic_con, topic_cur, len(context), len(cur)))
            turn_ids.append(torch.tensor(turn_id))
        

        topic_train.append(tokenizer(topic_data[i][0], truncation=True, max_length=512, padding=True, return_tensors='pt'))
        topic_num.append((len(topic_data[i][0]), topic_data[i][1]))
    
    # Padding
    id_inputs = pad_sequences(id_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    turn_ids = pad_sequences(turn_ids, maxlen=MAX_LEN, dtype="long", value=1, truncating="post", padding="post")
    

    topic_train_input = [i['input_ids'] for i in topic_train]
    topic_train_mask = [i['attention_mask'] for i in topic_train]
    
    # Attention masks
    attention_masks = []
    for sent in tqdm(id_inputs, desc="Creating attention masks"):
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    

    grouped_inputs, grouped_masks, grouped_topic, grouped_token_type_id = [], [], [], []
    count = 0
    for i in tqdm(sample_num_memory, desc="Grouping"):
        grouped_inputs.append(id_inputs[count: count+i])
        grouped_masks.append(attention_masks[count: count+i])
        grouped_topic.append(topic_inputs[count: count+i])
        grouped_token_type_id.append(turn_ids[count:count+i])
        count += i
    

    pos_neg_pairs, pos_neg_masks, pos_neg_token_types, topic_pairs = [], [], [], []
    topic_trains, topic_trains_mask, topic_nums = [], [], []
    
    for i in tqdm(range(len(grouped_inputs)), desc="Building pairs"):
        if len(grouped_inputs[i]) >= 2:

            pos_neg_pairs.append([grouped_inputs[i][0], grouped_inputs[i][1]])
            pos_neg_token_types.append([grouped_token_type_id[i][0], grouped_token_type_id[i][1]])
            pos_neg_masks.append([grouped_masks[i][0], grouped_masks[i][1]])
            
            topic_pairs.append([grouped_topic[i][0], grouped_topic[i][1]])
            topic_trains.append(topic_train_input[i])
            topic_trains_mask.append(topic_train_mask[i])
            topic_nums.append(topic_num[i])
            

            if len(grouped_inputs[i]) > 2:
                pos_neg_pairs.append([grouped_inputs[i][0], grouped_inputs[i][2]])
                pos_neg_token_types.append([grouped_token_type_id[i][0], grouped_token_type_id[i][2]])
                pos_neg_masks.append([grouped_masks[i][0], grouped_masks[i][2]])
                
                topic_pairs.append([grouped_topic[i][0], grouped_topic[i][2]])
                topic_trains.append(topic_train_input[i])
                topic_trains_mask.append(topic_train_mask[i])
                topic_nums.append(topic_num[i])
    

    train_inputs = torch.tensor(pos_neg_pairs)
    train_masks = torch.tensor(pos_neg_masks)
    train_types = torch.tensor(pos_neg_token_types)
    

    output_file = f'{args.output_dir}/processed_dialogue_pairs_{args.version}.pkl'
    pickle.dump((train_inputs, train_masks, train_types, topic_pairs, topic_trains, topic_trains_mask, topic_nums), 
                open(output_file, 'wb'))
    
    print(f"Data saved to {output_file}")
    print(f"Final dataset size: {len(train_inputs)} pairs")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--preference_data")
    parser.add_argument("--output_dir", default='./perference_data', help='Output directory for processed data')
    parser.add_argument("--version", default='dialogue_pairs', help='Version identifier for output files')
    parser.add_argument("--enhanced_negative", action='store_true', help='Use enhanced negative sampling strategy')
    parser.add_argument("--add_hard_negatives", action='store_true', help='Add hard negatives from other positive samples')
    parser.add_argument("--max_hard_negatives", type=int, default=2, help='Maximum number of hard negatives per positive sample')
    
    args = parser.parse_args()
    

    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
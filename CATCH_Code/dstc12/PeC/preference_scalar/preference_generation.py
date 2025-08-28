import torch
import numpy as np
import json
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from tqdm import tqdm
from datasets import load_dataset, Dataset
from proc_fuc import map_top, seg_create, add_seg, seg_create_new, add_seg_new, preproce_dataset
import logging
# from accelerate import Accelerator
from torch.utils.data import DataLoader



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('org_dataset', type=str)
    parser.add_argument('slice_data', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('save_path', type=str)
    return parser.parse_args()

def extract_dataset(seg_dataset, tokenizer, only_themed = True):
    if tokenizer.eos_token == None:
        eot = ""
        start_h = ""
        end_h = ""
    else:
        eot = '<|eot_id|>'
        start_h = '<|start_header_id|>'
        end_h = '<|end_header_id|>'

    if only_themed:
        themed_top_dataset = list()
        for conv in seg_dataset:
            for utt in conv:
                if utt['theme_label'] != None:
                    target_top = utt['start_top']
                    top_seq = ""
                    # exctract the same utt from the same topic and make them a sequence.
                    for utt in conv:
                        if utt['start_top'] == target_top:
                            top_seq = top_seq + start_h + utt['role'] + ':' + end_h + utt[utt['role']] + eot
                        elif utt['start_top'] > target_top:
                            break
                    themed_top_dataset.append(top_seq)
        return themed_top_dataset

    else:
        top_dataset = list()
        for conv in seg_dataset:
            for utt in conv:
                target_top = utt['start_top']
                top_seq = ""
                # exctract the same utt from the same topic and make them a sequence.
                for utt in conv:
                    if utt['start_top'] == target_top:
                        top_seq = top_seq + start_h + utt['role'] + ':' + end_h + utt[utt['role']] + eot
                    elif utt['start_top'] > target_top:
                        break
                top_dataset.append(top_seq)
        return top_dataset

def tokenize(example):
    data = example['conversations']
    tokenized_data = tokenizer(data, return_tensors="pt",truncation=True, max_length=512, padding=True)
    return {"input_ids" : tokenized_data['input_ids'].squeeze(0),
            "attention_mask": tokenized_data['attention_mask'].squeeze(0)
    }


# accelerator = Accelerator()
def main(model, dataset, slice_data, save_path):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model, num_labels = 1)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = torch.nn.DataParallel(model, device_ids=[0,1])
    model.to('cuda:0')
    # model = accelerator.prepare(model)
    model.eval()


    bound_ls = seg_create_new(slice_data)
    seg_dataset = add_seg_new(dataset, bound_ls)
    themed_top_dataset = extract_dataset(seg_dataset, tokenizer = tokenizer, only_themed=True)
    dataset = themed_top_dataset
    pair_dataset = map_top(dataset, pref_dataids = None, type="test", tokenizer = tokenizer)
    pair_dataset = Dataset.from_dict({'conversations':pair_dataset})
    tokenized_dataset = pair_dataset.map(tokenize, remove_columns = pair_dataset.column_names)


    batch_size = 100
    dataloader = DataLoader(
        tokenized_dataset,          
        batch_size=batch_size,   
        collate_fn = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=512, return_tensors="pt")
    )

    score_list = list()
    for batch in tqdm(dataloader):
        batch = {k: v.to("cuda:0") for k, v in batch.items()}
        try:
            with torch.no_grad():
                outputs = model(**batch)
            score_list.append(outputs.logits)
            logging.debug(outputs.logits)
        
        except:
            logging.debug("error occur")
            print("error occur")
            continue

    last_scores = score_list.pop()
    padding_size = batch_size - last_scores.shape[0]
    padded_tensor = torch.nn.functional.pad(last_scores, (0, 0, 0, padding_size), value=0)
    score_list.append(padded_tensor)
    score_matrix = torch.cat(score_list, dim =1).view(1,-1)

    torch.save(score_matrix, "./score_matrix_f1_post.pt")

    prob_matrix = (score_matrix- torch.min(score_matrix)) / (torch.max(score_matrix) - torch.min(score_matrix))
    prob_matrix = prob_matrix.squeeze(0)

    len_tops = len(themed_top_dataset)
    num_ls = [i for i in range(1,len_tops+1)]
    ind_ls = np.cumsum(num_ls)

    # initializer Col_list and make it as [[a11]]
    label_list = [prob_matrix[0:1]]
    # then add other col and make it become [[a11], [a12, a22], [a13,a23,a33],....]
    for i in range(1,len(ind_ls)):
        col = prob_matrix[ind_ls[i-1]:ind_ls[i]]
        label_list.append(col)

    should_list = list()
    cannot_list = list()
    for i, data in tqdm(enumerate(label_list)):
        for j, label in enumerate(data):
            # ignore the diagonal element
            if i == j:
                continue
            elif label > 0.90:
                utt_ids_1 = themed_top_dataset[i]
                utt_ids_2 = themed_top_dataset[j]
                for id_1 in utt_ids_1:
                    for id_2 in utt_ids_2:
                        should_list.append([id_1, id_2])
            elif label < 0.1:
                utt_ids_1 = themed_top_dataset[i]
                utt_ids_2 = themed_top_dataset[j]
                for id_1 in utt_ids_1:
                    for id_2 in utt_ids_2:
                        cannot_list.append([id_1, id_2])
                

    print(len(should_list))
    print(len(cannot_list))

    pref_datapairs = {"should_link": should_list,
                    "cannot_link": cannot_list}

    with open(save_path, "w") as f:
        json.dump(pref_datapairs, f, indent=4)
        
        
        
        
        
if __name__ == '__main__':
    args = parse_args()
    
    with open(args.org_dataset) as fw:
        org_dataset = [json.loads(line) for line in f]
    dataset = preproce_dataset(org_dataset) 

    with open(args.slice_data) as fw:
    # slice_data = fw.read()
        slice_data = [json.loads(line) for line in fw]
        
        
    main(args.model, dataset, slice_data, args.save_path)





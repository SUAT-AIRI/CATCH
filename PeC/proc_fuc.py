import json
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset, Dataset
import re
import torch

# tokenizer = AutoTokenizer.from_pretrained("/home/wangkuang/workshop/git/Meta-Llama-3-8B-Instruct")
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "left"

def preproce_dataset(dataset):
    pro_dataset_0 = list()
    pro_dataset_1 = list()
    for i in tqdm(range(0, len(dataset))):
        data = dataset[i]
        new_cov_0, new_cov_1 = list(), list()
        for utt in data['turns']:
            e0 = {'role':utt["speaker_role"],
                utt["speaker_role"]: utt['utterance'],
                "theme_label": utt['theme_label'],
                "utterance_id":utt['utterance_id']}
            new_cov_0.append(e0) 
            e1 = {utt["speaker_role"]: utt['utterance']}
            new_cov_1.append(e1)

        pro_dataset_0.append(new_cov_0)
        pro_dataset_1.append(new_cov_1)
    return pro_dataset_0

def map_utt(dataset, pref_dataids, type, tokenizer):
    if tokenizer.eos_token == None:
        eos = ""
        eot = ""
    else:
        eot = '<|eot_id|>'
        eos = tokenizer.eos_token

    if type == "train":
        rw_dataset = list()
        for pair in tqdm(pref_dataids):
            question_id = pair[0]
            accept_id = pair[1]
            reject_id = pair[2]

            for data in dataset:
                for utt in data:
                    if utt["utterance_id"] == question_id:
                        question = "Refer to: \"" + utt[utt['role']]  + "\" , consider whether the following sentence have the same topic:"
                    elif utt["utterance_id"] == accept_id:
                        chosen = question + utt[utt['role']] + eot + eos
                    elif utt['utterance_id'] == reject_id:
                        reject = question + utt[utt['role']] + eot + eos

            chosen_dict = tokenizer(chosen, return_attention_mask=True)
            reject_dict = tokenizer(reject, return_attention_mask=True)
            input_ids_chosen = chosen_dict['input_ids']
            attention_mask_chosen = chosen_dict['attention_mask']
            input_ids_rejected = reject_dict['input_ids']
            attention_mask_rejected = reject_dict['attention_mask']

            n_data = {"input_ids_chosen" :input_ids_chosen,
                    "attention_mask_chosen" : attention_mask_chosen,
                    "input_ids_rejected" : input_ids_rejected,
                    "attention_mask_rejected" : attention_mask_rejected}
            rw_dataset.append(n_data)
        return rw_dataset


    if type == "struct_test":
        # create a dataset that contains all single sentence as its element (i.e. lis_dataset[i])
        lis_dataset = list()
        for data in dataset:
            for utt in data:
                lis_dataset.append(utt)
    
        # test_dataset is a right_upper_triangle_matrix[[a11][a12,a22],[a13,a23,a33]...]
        #                                                col1    col2      col3
        test_dataset = list()
        qst_dataset = list()
        for utt in tqdm(lis_dataset):
            question = "Refer to: \"" + utt[utt['role']]  + "\" , consider whether the following sentence have the same topic:"
            qst_dataset.append(question)
            col = list()
            for qst in qst_dataset:
                seq = qst + utt[utt['role']] + eot + eos
                col.append(seq)
            token_col = tokenizer(col, return_tensors="pt", padding=True)
            test_dataset.append(token_col)
        return test_dataset


    if type == "test":
        # create a dataset that contains all single sentence as its element (i.e. lis_dataset[i])
        lis_dataset = list()
        for data in dataset:
            for utt in data:
                lis_dataset.append(utt)
    
        # pair_dataset is a right_upper_triangle_matrix[a11,  a12,a22,   a13,a23,a33,  ...]
        #                                                col1    col2      col3
        test_dataset = list()
        qst_dataset = list()
        pair_dataset = list()
        for utt in tqdm(lis_dataset):
            question = "Refer to: \"" + utt[utt['role']]  + "\" , consider whether the following sentence have the same topic:"
            qst_dataset.append(question)
            for qst in qst_dataset:
                seq = qst + utt[utt['role']] + eot + eos
                pair_dataset.append(seq)
        # pair_dataset = Dataset.from_dict({'conversations':pair_dataset})
        # tokenized_dataset = pair_dataset.map(map_fuc)
        # print(pair_dataset)
        
        
        return pair_dataset


def map_top(seg_dataset, pref_dataids, type, tokenizer):
    if tokenizer.eos_token == None:
        eos = ""
        eot = ""
        start_h = ""
        end_h = ""
    else:
        eot = '<|eot_id|>'
        eos = tokenizer.eos_token
        start_h = '<|start_header_id|>'
        end_h = '<|end_header_id|>'

    if type == "train_v2":
        rw_dataset = list()
        for pair in tqdm(pref_dataids):
            question_id = pair[0]
            accept_id_1 = pair[1][-1]
            accept_id_2 = pair[2][-1]
            reject_id = pair[3][-1]

            for data in seg_dataset:
                for utt in data:
                    if utt["utterance_id"] == question_id:
                        target_top = utt['start_top']
                        quest_top = ""
                        for utt in data:
                            if utt['start_top'] == target_top:
                                quest_top = quest_top + start_h + utt['role'] + ':' + end_h + utt[utt['role']] + eot
                            elif utt['start_top'] > target_top:
                                break
                        question = "Refer to: \"" + quest_top  + "\" , consider whether the following set of utterances have the same topic:"
                    
                    elif utt["utterance_id"] == accept_id_1:
                        target_top = utt['start_top']
                        accept_top = ""
                        for utt in data:
                            if utt['start_top'] == target_top:
                                accept_top = accept_top + start_h + utt['role'] + ':' + end_h + utt[utt['role']] + eot
                            elif utt['start_top'] > target_top:
                                break
                        chosen_1 = accept_top + eot + eos

                    elif utt["utterance_id"] == accept_id_2:
                        target_top = utt['start_top']
                        accept_top = ""
                        for utt in data:
                            if utt['start_top'] == target_top:
                                accept_top = accept_top + start_h + utt['role'] + ':' + end_h + utt[utt['role']] + eot
                            elif utt['start_top'] > target_top:
                                break
                        chosen_2 = accept_top + eot + eos

                    elif utt['utterance_id'] == reject_id:
                        target_top = utt['start_top']
                        reject_top = ""
                        for utt in data:
                            if utt['start_top'] == target_top:
                                reject_top = reject_top + start_h + utt['role'] + ':' + end_h + utt[utt['role']] + eot
                            elif utt['start_top'] > target_top:
                                break
                        reject = reject_top + eot + eos

            quset_chosen_1 = question + chosen_1
            quset_chosen_2 = question + chosen_2
            quest_reject = question + reject

            chosen_dict_1 = tokenizer(quset_chosen_1, return_attention_mask=True, max_length=512, truncation=True)
            reject_dict = tokenizer(quest_reject, return_attention_mask=True, max_length=512, truncation=True)
            input_ids_chosen = chosen_dict_1['input_ids']
            attention_mask_chosen = chosen_dict_1['attention_mask']
            input_ids_rejected = reject_dict['input_ids']
            attention_mask_rejected = reject_dict['attention_mask']

            n_data_1 = {"input_ids_chosen" :input_ids_chosen,
                    "attention_mask_chosen" : attention_mask_chosen,
                    "input_ids_rejected" : input_ids_rejected,
                    "attention_mask_rejected" : attention_mask_rejected}
            rw_dataset.append(n_data_1)


            chosen_dict_2 = tokenizer(quset_chosen_2, return_attention_mask=True, max_length=512, truncation=True)
            input_ids_chosen = chosen_dict_2['input_ids']
            attention_mask_chosen = chosen_dict_2['attention_mask']
            
            n_data_2 = {"input_ids_chosen" :input_ids_chosen,
                    "attention_mask_chosen" : attention_mask_chosen,
                    "input_ids_rejected" : input_ids_rejected,
                    "attention_mask_rejected" : attention_mask_rejected}
            rw_dataset.append(n_data_2)
        return rw_dataset

    if type == "train":
        rw_dataset = list()
        for pair in tqdm(pref_dataids):
            question_id = pair[0]
            accept_id = pair[1]
            reject_id = pair[2]

            for data in seg_dataset:
                for utt in data:
                    if utt["utterance_id"] == question_id:
                        target_top = utt['start_top']
                        quest_top = ""
                        for utt in data:
                            if utt['start_top'] == target_top:
                                quest_top = quest_top + start_h + utt['role'] + ':' + end_h + utt[utt['role']] + eot
                            elif utt['start_top'] > target_top:
                                break
                        question = "Refer to: \"" + quest_top  + "\" , consider whether the following set of utterances have the same topic:"
                    
                    elif utt["utterance_id"] == accept_id:
                        target_top = utt['start_top']
                        accept_top = ""
                        for utt in data:
                            if utt['start_top'] == target_top:
                                accept_top = accept_top + start_h + utt['role'] + ':' + end_h + utt[utt['role']] + eot
                            elif utt['start_top'] > target_top:
                                break
                        chosen = question + accept_top + eot + eos

                    elif utt['utterance_id'] == reject_id:
                        target_top = utt['start_top']
                        reject_top = ""
                        for utt in data:
                            if utt['start_top'] == target_top:
                                reject_top = reject_top + start_h + utt['role'] + ':' + end_h + utt[utt['role']] + eot
                            elif utt['start_top'] > target_top:
                                break
                        reject = question + reject_top + eot + eos

            chosen_dict = tokenizer(chosen, return_attention_mask=True, max_length=512, truncation=True)
            reject_dict = tokenizer(reject, return_attention_mask=True, max_length=512, truncation=True)
            input_ids_chosen = chosen_dict['input_ids']
            attention_mask_chosen = chosen_dict['attention_mask']
            input_ids_rejected = reject_dict['input_ids']
            attention_mask_rejected = reject_dict['attention_mask']

            n_data = {"input_ids_chosen" :input_ids_chosen,
                    "attention_mask_chosen" : attention_mask_chosen,
                    "input_ids_rejected" : input_ids_rejected,
                    "attention_mask_rejected" : attention_mask_rejected}
            rw_dataset.append(n_data)
        return rw_dataset

    if type == "finetune":
        rw_dataset = list()
        should_pairs = pref_dataids['should_link']
        for pair in tqdm(should_pairs):
            question_id = pair[0]
            accept_id = pair[1]

            for data in seg_dataset:
                for utt in data:
                    if utt["utterance_id"] == question_id:
                        target_top = utt['start_top']
                        quest_top = ""
                        for utt in data:
                            if utt['start_top'] == target_top:
                                quest_top = quest_top + start_h + utt['role'] + ':' + end_h + utt[utt['role']] + eot
                            elif utt['start_top'] > target_top:
                                break
                        question = "Refer to: \"" + quest_top  + "\" , consider whether the following set of utterances have the same topic:"
                    
                    elif utt["utterance_id"] == accept_id:
                        target_top = utt['start_top']
                        accept_top = ""
                        for utt in data:
                            if utt['start_top'] == target_top:
                                accept_top = accept_top + start_h + utt['role'] + ':' + end_h + utt[utt['role']] + eot
                            elif utt['start_top'] > target_top:
                                break
                        chosen = question + accept_top + eot + eos


            chosen_dict = tokenizer(chosen, return_attention_mask=True, max_length=512, truncation=True)
            input_ids_chosen = chosen_dict['input_ids']
            attention_mask_chosen = chosen_dict['attention_mask']
            label = torch.tensor([1], dtype=torch.float32)

            n_data = {"input_ids" :input_ids_chosen,
                    "attention_mask" : attention_mask_chosen,
                    "labels": label}
            rw_dataset.append(n_data)
            
        cannot_pairs = pref_dataids['cannot_link']
        for pair in tqdm(cannot_pairs):
            question_id = pair[0]
            reject_id = pair[1]

            for data in seg_dataset:
                for utt in data:
                    if utt["utterance_id"] == question_id:
                        target_top = utt['start_top']
                        quest_top = ""
                        for utt in data:
                            if utt['start_top'] == target_top:
                                quest_top = quest_top + start_h + utt['role'] + ':' + end_h + utt[utt['role']] + eot
                            elif utt['start_top'] > target_top:
                                break
                        question = "Refer to: \"" + quest_top  + "\" , consider whether the following set of utterances have the same topic:"
                
                    elif utt['utterance_id'] == reject_id:
                        target_top = utt['start_top']
                        reject_top = ""
                        for utt in data:
                            if utt['start_top'] == target_top:
                                reject_top = reject_top + start_h + utt['role'] + ':' + end_h + utt[utt['role']] + eot
                            elif utt['start_top'] > target_top:
                                break
                        reject = question + reject_top + eot + eos

           
            reject_dict = tokenizer(reject, return_attention_mask=True, max_length=512, truncation=True)
            input_ids_rejected = reject_dict['input_ids']
            attention_mask_rejected = reject_dict['attention_mask']
            label = torch.tensor([0], dtype=torch.float32)
            n_data = {
                    "input_ids" : input_ids_rejected,
                    "attention_mask" : attention_mask_rejected,
                    "labels" : label}
            rw_dataset.append(n_data)
            
        return rw_dataset
    

    if type == "struct_test":
        # test_dataset is a right_upper_triangle_matrix[[a11][a12,a22],[a13,a23,a33]...]
        #                                                col1    col2      col3
        test_dataset = list()
        qst_dataset = list()
        for top in tqdm(seg_dataset):
            question = "Refer to: \"" + top  + "\" , consider whether the following set of utterances have the same topic:"
            qst_dataset.append(question)
            col = list()
            for qst in qst_dataset:
                seq = qst + top + eot + eos
                col.append(seq)
            token_col = tokenizer(col, return_tensors="pt", padding=True)
            test_dataset.append(token_col)
        return test_dataset


    if type == "test":
        # pair_dataset is a right_upper_triangle_matrix[a11,  a12,a22,   a13,a23,a33,  ...]
        #                                                col1    col2      col3
        test_dataset = list()
        qst_dataset = list()
        pair_dataset = list()
        for top in tqdm(seg_dataset):
            question = "Refer to: \"" + top  + "\" , consider whether the following set of utterances have the same topic:"
            qst_dataset.append(question)
            for qst in qst_dataset:
                seq = qst + top + eot + eos
                pair_dataset.append(seq)
        # pair_dataset = Dataset.from_dict({'conversations':pair_dataset})
        # tokenized_dataset = pair_dataset.map(map_fuc)
        # print(pair_dataset)
        return pair_dataset

def seg_create(slice_data):
    pattern_ind = r'\b\d+/\d+\b'
    slices = re.split(pattern_ind, slice_data)

    bound_ls = list()
    pattern_tar = r'\[(?:\d+,\s*)+\d+\]'
    for i in tqdm(range(len(slices[1:]))):
        tar_slice = re.findall(pattern_tar, slices[i])
        if tar_slice == []:
            bounds_n = None   
        else:
            bounds = tar_slice[0].split(']')[0].split('[')[1]
            bounds_n = [int(i) for i in re.findall(r'\d+',bounds)]
        tar_elm = [i, bounds_n]
        bound_ls.append(tar_elm)

    bound_ls = bound_ls[1:-1]
    return bound_ls

def seg_create_new(slice_data):   
    bound_ls = list()
    for i, top_seg in enumerate(tqdm(slice_data)):
        ind = i + 1
        top_seg[0] = top_seg[0] + 1
        tar_elm = [ind, top_seg]
        bound_ls.append(tar_elm)
    return bound_ls

def add_seg(dataset, bound_ls):
    for i in tqdm(range(0,len(dataset))):
        bound = bound_ls[i][1]
        data = dataset[i]
        
        for element in data:
            element['start_top'] = 0
        if bound == None:
            continue
        e = 0
        top_ind = 1
        data[e]['start_top'] = 1
        for b in bound:
            last_e = e
            e = e + b
            for j in range(last_e, e+1):
                data[j]['start_top'] = top_ind
            top_ind = top_ind + 1
    return dataset

def add_seg_new(dataset, bound_ls):
    for i in tqdm(range(0,len(dataset))):
        bound = bound_ls[i][1]
        data = dataset[i]
        
        for element in data:
            element['start_top'] = 0
        if bound == None:
            continue
        e = 0
        top_ind = 1
        data[e]['start_top'] = 1
        for b in bound:
            last_e = e
            e = e + b
            for j in range(last_e, e):
                data[j]['start_top'] = top_ind
            top_ind = top_ind + 1
    return dataset
        
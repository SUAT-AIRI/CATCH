import os
import re
import copy
import json
import torch
import pickle
import segeval
import argparse
import numpy as np
from tqdm import tqdm
from model import SegModel
from torch.nn.utils.rnn import pad_sequence
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForNextSentencePrediction, AutoTokenizer, set_seed

DATASET = {'doc': 'doc2dial', '711': 'dialseg711', 'TIAGE': 'TIAGE'}


def depth_score_cal(scores):
    output_scores = []
    for i in range(len(scores)):
        lflag, rflag = scores[i], scores[i]
        if i == 0:
            hl = scores[i]
            for r in range(i + 1, len(scores)):
                if rflag <= scores[r]:
                    rflag = scores[r]
                else:
                    break
        elif i == len(scores) - 1:
            hr = scores[i]
            for l in range(i - 1, -1, -1):
                if lflag <= scores[l]:
                    lflag = scores[l]
                else:
                    break
        else:
            for r in range(i + 1, len(scores)):
                if rflag <= scores[r]:
                    rflag = scores[r]
                else:
                    break
            for l in range(i - 1, -1, -1):
                if lflag <= scores[l]:
                    lflag = scores[l]
                else:
                    break
        depth_score = 0.5 * (lflag + rflag - 2 * scores[i])
        output_scores.append(depth_score)

    return output_scores



def dstc_infer(args, model_path):
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=True)

    model = SegModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), False)
    model.to(args.device)
    model.eval()
    res = {}
    threshold_scores = []
    all_seg_p = []
    with open(args.dataset, 'r') as f1:
        test_dataset = []
        test_labels = []
        for line in f1:
            dial_data = json.loads(line)
            dials = []
            theme_flag = []
            for turn in dial_data['turns']:
                dials.append(turn['utterance'])
                if turn['theme_label']:
                    theme_flag.append(1)
                else:
                    theme_flag.append(0)
            test_dataset.append(dials[1:])
            test_labels.append(theme_flag[1:])
    
    c = score_wd = score_pk = 0
    for index in tqdm(range(len(test_dataset))):
        id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [[],
                                                                                                      []], [
                                                                                             [], []], [[], []]
        depth_scores = []
        seg_r_labels, seg_r = [], []
        tmp = 0
        text = test_dataset[index]
        print("测试集长度：", len(text), flush=True)
        tmps = test_labels[index]
        for t in tmps:
            if int(t) == 0:
                tmp += 1
            else:
                if tmp > 0:
                    seg_r.append(tmp)
                tmp = 1
        seg_r.append(tmp)
        for i in range(len(text) - 1):
            context, cur = [], []
            l, r = i, i + 1
            for win in range(args.window_size):
                if l > -1:
                    context.append(text[l][:128])
                    l -= 1
                if r < len(text):
                    cur.append(text[r][:128])
                    r += 1
            context.reverse()

            topic_con = tokenizer(context, truncation=True, padding=True, max_length=256, return_tensors='pt')
            topic_cur = tokenizer(cur, truncation=True, padding=True, max_length=256, return_tensors='pt')

            topic_input[0].extend(topic_con['input_ids'])
            topic_input[1].extend(topic_cur['input_ids'])
            topic_att_mask[0].extend(topic_con['attention_mask'])
            topic_att_mask[1].extend(topic_cur['attention_mask'])
            topic_num[0].append(len(context))
            topic_num[1].append(len(cur))

            sent1 = ''
            for sen in context:
                sent1 += sen + '[SEP]'

            sent2 = text[i + 1]

            encoded_sent1 = tokenizer.encode(sent1, add_special_tokens=True, max_length=256, return_tensors='pt')
            encoded_sent2 = tokenizer.encode(sent2, add_special_tokens=True, max_length=256, return_tensors='pt')
            encoded_pair = encoded_sent1[0].tolist()[:-1] + encoded_sent2[0].tolist()[1:]
            type_id = [0] * len(encoded_sent1[0].tolist()[:-1]) + [1] * len(encoded_sent2[0].tolist()[1:])
            type_ids.append(torch.Tensor(type_id))
            id_inputs.append(torch.Tensor(encoded_pair))

        MAX_LEN = 512
        id_inputs = pad_sequences(id_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post",
                                  padding="post")
        type_ids = pad_sequences(type_ids, maxlen=MAX_LEN, dtype="long", value=1, truncating="post", padding="post")
        for sent in id_inputs:
            att_mask = [int(token_id > 0) for token_id in sent]
            coheren_att_masks.append(att_mask)

        try:
            topic_input = [pad_sequence(i, batch_first=True).to(args.device) for i in topic_input]
            topic_mask = [pad_sequence(i, batch_first=True).to(args.device) for i in topic_att_mask]
        except:
            print(file)
            continue

        with torch.no_grad():
            coheren_inputs = torch.tensor(id_inputs).to(args.device)
            coheren_masks = torch.tensor(coheren_att_masks).to(args.device)
            coheren_type_ids = torch.tensor(type_ids).to(args.device)
            scores = model.infer(coheren_inputs, coheren_masks, coheren_type_ids, topic_input, topic_mask,
                                 topic_num)

        depth_scores = depth_score_cal(scores)

        # boundary_indice = np.argsort(np.array(depth_scores))[-args.pick_num:]
        picked_index = []
        seg_number = 0
        for segg_r in seg_r:
            seg_number += segg_r
            picked_index.append(seg_number)
        threshold_scores += [depth_scores[s] for s in picked_index[:-1]]
        # print(len(threshold_scores), flush=True)
        # threshold = min([depth_scores[i] for i in])
        threshold = 0.5
        boundary_indice = np.where(np.array(depth_scores) > threshold)[0]


        seg_p_labels = [0] * (len(depth_scores) + 1)
        for i in boundary_indice:
            seg_p_labels[i] = 1

        tmp = 0
        seg_p = []
        for fake in seg_p_labels:
            if fake == 1:
                tmp += 1
                seg_p.append(tmp)
                tmp = 0
            else:
                tmp += 1
        seg_p.append(tmp)
        print(seg_p, seg_r, flush=True)
        all_seg_p.append(seg_p)
        score_wd += segeval.window_diff(seg_p, seg_r)
        score_pk += segeval.pk(seg_p, seg_r)

        c += 1

    print('pk: ', score_pk / c)
    print('wd: ', score_wd / c)


    with open(args.save_name, 'w') as f2:
        for i in all_seg_p:
            f2.write(str(i))
            f2.write("\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--save_name", default='epoch')

    parser.add_argument("--single_ckpt", action='store_true')
    parser.add_argument("--ckpt")
    parser.add_argument("--root", default='.')
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--ckpt_start", type=int, default=0)
    parser.add_argument("--ckpt_end", type=int, default=3)
    parser.add_argument("--pick_num", type=int, default=6)
    parser.add_argument("--window_size", default=2, type=int)

    args = parser.parse_args()
    assert args.single_ckpt and args.ckpt
    args.dataset = DATASET[args.dataset]
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    os.makedirs(f'{args.root}/metric/{args.model}', exist_ok=True)
    set_seed(3407)

    PATH = 'model/dstc_perference/5'
    dstc_infer(args, PATH)
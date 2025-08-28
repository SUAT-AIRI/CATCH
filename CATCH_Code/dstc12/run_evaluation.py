# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0


from pathlib import Path
import sys
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ⇩⇩⇩ 把 <项目根>/src 加到 sys.path 最前面
ROOT = Path(__file__).resolve().parents[1]   # scripts/.. → 项目根
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
# ⇧⇧⇧

from argparse import ArgumentParser
import json
import os

import getpass
from langchain.embeddings import HuggingFaceEmbeddings
import src.patches.coclust_external_fix    # 必须放在 import acc 之前
from dstc12.eval import (
    acc,
    nmi,
    rouge_with_multiple_references,
    cosine_similarity_with_multiple_references,
    llm_score
)
from dstc12.utils import get_llm
import logging

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
level=logging.DEBUG,
filename='eval_log_self.log',
filemode='a')


def run_non_llm_eval(references, predictions, embedding_model_name):
    label_1_references, label_2_references = references
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    reference_1_embeddings = embeddings.embed_documents(label_1_references)
    reference_2_embeddings = embeddings.embed_documents(label_2_references)
    predictions_embeddings = embeddings.embed_documents(predictions)

    avg_acc = acc(references=label_1_references, predictions=predictions)
    avg_nmi = nmi(references=label_1_references, predictions=predictions)
    avg_rouge = rouge_with_multiple_references(
        [[label_1, label_2] for label_1, label_2 in zip(label_1_references, label_2_references)],
        predictions
    )
    avg_cosine_similarity = cosine_similarity_with_multiple_references(
        (reference_1_embeddings, reference_2_embeddings),
        predictions_embeddings
    )
    return {
        'acc': avg_acc,
        'nmi': avg_nmi,
        'rouge_1': avg_rouge['rouge1'].fmeasure,
        'rouge_2': avg_rouge['rouge2'].fmeasure,
        'rouge_l': avg_rouge['rougeL'].fmeasure,
        'cosine_similarity': avg_cosine_similarity,
    }


def run_llm_eval(predictions, llm_name):
    llm = get_llm(llm_name)
    avg_llm_score = llm_score(predictions, llm)
    return {
        'llm_score': avg_llm_score,
    }
    # return 0




def main(references, predictions, embedding_model_name, llm_name):
    non_llm_metrics = run_non_llm_eval(references, predictions, embedding_model_name)
    logging.debug(non_llm_metrics)
    llm_metrics = run_llm_eval(predictions, llm_name)
    logging.debug(llm_metrics)
    return non_llm_metrics | llm_metrics


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('ground_truth_file', type=str)
    parser.add_argument('predictions_file', type=str)
    parser.add_argument('--embedding-model-name', type=str)
    parser.add_argument('--llm-name', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.ground_truth_file) as f:
        ground_truth = [json.loads(line) for line in f]
    with open(args.predictions_file) as f:
        predictions = [json.loads(line) for line in f]

    label1_references, label2_references, label_predictions = [], [], []
    for dialog_gt, dialog_pred in zip(ground_truth, predictions):
        assert len(dialog_gt['turns']) == len(dialog_pred['turns'])
        for utterance_gt, utterance_pred in zip(dialog_gt['turns'], dialog_pred['turns']):
            assert utterance_gt['utterance_id'] == utterance_pred['utterance_id']
            if utterance_gt['theme_label'] is None:
                continue
            uid = utterance_gt['utterance_id']
            label1_references.append(utterance_gt['theme_label']['label_1'])
            label2_references.append(utterance_gt['theme_label']['label_2'])
            label_predictions.append(utterance_pred['theme_label_predicted'])
    metrics = main((label1_references, label2_references), label_predictions, args.embedding_model_name, args.llm_name)
    for metric, value in metrics.items():
        print(f'{metric}: {value:.3f}')
    logging.debug(metrics)

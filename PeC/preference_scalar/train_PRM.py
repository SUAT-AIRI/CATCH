from datasets import Dataset
from argparse import ArgumentParser
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, get_peft_model, TaskType
import json
from tqdm import tqdm
from proc_fuc import map_top, seg_create, add_seg, seg_create_new, add_seg_new, preproce_dataset

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('save_path', type=str)
    parser.add_argument('org_dataset', type=str)
    parser.add_argument('slice_file', type=str)
    parser.add_argument('pref_file', type=str)
    return parser.parse_args()

def main(model_path, save_path, dataset, slice_data, pref_data):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
    model.train()


    bound_ls = seg_create_new(slice_data)
    seg_dataset = add_seg_new(dataset, bound_ls)
    tokenized_dataset = map_top(seg_dataset, pref_dataids, type="finetune", tokenizer=tokenizer)
    rw_dataset = Dataset.from_list(tokenized_dataset)


    training_args = RewardConfig(
        output_dir= save_path,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        save_strategy='epoch',
        save_total_limit=1,
        num_train_epochs=5,
        logging_steps=10,
        save_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=rw_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.save_model( save_path)




if __name__ == '__main__':
    args = parse_args()

    # if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        # os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")
        
        
    with open(args.org_dataset) as f:
        org_dataset = [json.loads(line) for line in f]
    dataset = preproce_dataset(org_dataset)
        
    with open(args.slice_file) as f:
        slice_data = [json.loads(line) for line in f]
    with open(args.pref_file) as prefs_in:
        pref_dataids = json.load(prefs_in)

        
    cluster_label_map = main(
        args.model_path,
        args.save_path,
        dataset,
        slice_data,
        pref_dataids,
    )
                
    with open(args.result_file, 'w') as fw:
        json.dumps(cluster_label_map, fw)

import re
import os
import json
import torch
import random
import pickle
import argparse
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from model import SegModel
from torch.cuda import amp
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertForNextSentencePrediction, BertConfig, get_linear_schedule_with_warmup, set_seed, \
    AutoModel

DATASET = {'doc': 'doc2dial', '711': 'dialseg711', 'TIAGE': 'TIAGE'}


def get_mask(tensor):
    attention_masks = []
    for sent in tensor:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return torch.tensor(attention_masks)


class ourdataset(Dataset):
    def __init__(self, loaded_data):
        self.loaded_data = loaded_data

    def __getitem__(self, idx):
        return [i[idx] for i in self.loaded_data]

    def __len__(self):
        return len(self.loaded_data[0])

    def collect_fn(self, examples):
        batch_size, topic_train, topic_train_mask, topic_num = len(examples), torch.tensor(0), torch.tensor(
            0), torch.tensor(0)
        coheren_inputs = pad_sequence([ex[0] for ex in examples], batch_first=True)
        coheren_mask = pad_sequence([ex[1] for ex in examples], batch_first=True)
        coheren_type = pad_sequence([ex[2] for ex in examples], batch_first=True)

        topic_context = pad_sequence([torch.tensor(j) for ex in examples for j in ex[3][0][0]['input_ids']],
                                     batch_first=True)
        topic_pos = pad_sequence([torch.tensor(j) for ex in examples for j in ex[3][0][1]['input_ids']],
                                 batch_first=True)
        topic_neg = pad_sequence([torch.tensor(j) for ex in examples for j in ex[3][1][1]['input_ids']],
                                 batch_first=True)

        topic_context_num = [ex[3][0][2] for ex in examples]
        topic_pos_num = [ex[3][0][3] for ex in examples]
        topic_neg_num = [ex[3][1][3] for ex in examples]

        topic_context_mask, topic_pos_mask, topic_neg_mask = get_mask(topic_context), get_mask(topic_pos), get_mask(
            topic_neg)

        topic_train = pad_sequence([j for ex in examples for j in ex[4]], batch_first=True)
        topic_train_mask = pad_sequence([j for ex in examples for j in ex[5]], batch_first=True)
        topic_num = [ex[6] for ex in examples]

        return coheren_inputs, coheren_mask, coheren_type, topic_context, topic_pos, topic_neg, \
               topic_context_mask, topic_pos_mask, topic_neg_mask, \
               topic_context_num, topic_pos_num, topic_neg_num, \
               topic_train, topic_train_mask, topic_num


def setup():
    """
    Initialize the distributed training environment using torchrun
    """
    # 使用torchrun时，这些环境变量会自动设置
    dist.init_process_group(backend="nccl")


def cleanup():
    """
    Clean up the distributed training environment
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def main(args):
    # Setup the distributed environment
    setup()
    
    # 获取rank和world_size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set the device
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print("Start Training")
        print(f"World size: {world_size}")
    
    # 检查数据文件是否存在
    # data_path = '../perference_data/processed_dialogue_pairs.pkl'
    data_path = '../perference_data/processed_dialogue_pairs.pkl'
    if not os.path.exists(data_path):
        if rank == 0:
            print(f"Error: Data file not found: {data_path}")
        return {}
    
    # Load data
    if rank == 0:
        print(f"Loading data from {data_path}")
    
    try:
        loaded_data = pickle.load(open(data_path, 'rb'))
    except Exception as e:
        if rank == 0:
            print(f"Error loading data: {e}")
        return {}

    # Prepare dataset
    train_data = ourdataset(loaded_data)

    print("ALL Length: ", len(train_data))
    # Create distributed sampler
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)

    # Create dataloader with distributed sampler
    train_dataloader = DataLoader(train_data, 
                                  sampler=train_sampler, 
                                  batch_size=args.batch_size, 
                                  collate_fn=train_data.collect_fn,
                                  pin_memory=True,
                                  num_workers=2)  # 减少worker数量

    # Initialize model
    try:
        model = SegModel(margin=args.margin, train_split=args.train_split, window_size=args.window_size).to(device)
        
        # 检查模型文件是否存在
        model_path = "model/dstc/3"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device), False)
            if rank == 0:
                print(f"Loaded model from {model_path}")
        else:
            if rank == 0:
                print(f"Warning: Model file not found: {model_path}")
    except Exception as e:
        if rank == 0:
            print(f"Error initializing model: {e}")
        return {}
    
    # Wrap model with DistributedDataParallel
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # Resume from checkpoint if specified
    if args.resume and args.ckpt:
        ckpt_path = f'{args.root}/model/{args.ckpt}'
        if os.path.exists(ckpt_path):
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            model.load_state_dict(torch.load(ckpt_path, map_location=map_location), False)
            if rank == 0:
                print(f"Resumed from checkpoint: {ckpt_path}")

    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    
    # Calculate total steps
    epochs = args.epoch
    total_steps = len(train_dataloader) * epochs
    num_warmup_steps = int(total_steps * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    # Gradient scaler for mixed precision training
    scaler = amp.GradScaler(enabled=not args.no_amp)

    # Training loop
    for epoch_i in range(epochs):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch_i)

        # Only print for the main process
        if rank == 0:
            print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
            progress_bar = tqdm(range(len(train_dataloader)), desc="Training Progress")

        total_loss = 0
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            # Prepare input data
            input_data = {
                'coheren_inputs': batch[0].to(device),
                'coheren_mask': batch[1].to(device),
                'coheren_type': batch[2].to(device),
                'topic_context': batch[3].to(device),
                'topic_pos': batch[4].to(device),
                'topic_neg': batch[5].to(device),
                'topic_context_mask': batch[6].to(device),
                'topic_pos_mask': batch[7].to(device),
                'topic_neg_mask': batch[8].to(device),
                'topic_context_num': batch[9],
                'topic_pos_num': batch[10],
                'topic_neg_num': batch[11],
                'topic_train': batch[12].to(device),
                'topic_train_mask': batch[13].to(device),
                'topic_num': batch[14]
            }

            # Zero gradients
            model.zero_grad()

            # Forward pass with mixed precision
            try:
                with amp.autocast(enabled=not args.no_amp):
                    loss, margin_loss, topic_loss = model(input_data, args.window_size)
            except Exception as e:
                if rank == 0:
                    print(f"Error in forward pass: {e}")
                continue

            # Average loss across GPUs
            loss = loss.mean()

            # Accumulate total loss
            if loss.item() < 10:
                total_loss += loss.item()

            # Backward pass with gradient scaling
            if not args.no_amp:
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Step scheduler
            scheduler.step()
            if rank == 0:
                progress_bar.update(1)
                if step % 1000 == 0:  # 减少输出频率
                    print(f"Step {step}, loss={loss.item():.4f}")

        # Average loss and save model for main process
        avg_train_loss = total_loss / max(len(train_dataloader), 1)
        if rank == 0:
            print(f'=========== Loss for epoch {epoch_i}: {avg_train_loss}')
            
            # Save model
            PATH = f'model/dstc_perference/{epoch_i}'
            os.makedirs(os.path.dirname(PATH), exist_ok=True)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), PATH)
            progress_bar.close()

    # Cleanup distributed environment
    cleanup()
    return {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='dstc12')
    parser.add_argument("--save_model_name", default='dstc_perference')
    # model parameters
    parser.add_argument("--margin", type=int, default=1)
    parser.add_argument("--train_split", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=5)

    # path parameters
    parser.add_argument("--ckpt")
    parser.add_argument("--data_name", default='')
    parser.add_argument("--root", default='.')
    parser.add_argument("--epoch", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3407)

    # train parameters
    parser.add_argument('--accum', type=int, default=1)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)

    # device parameters
    parser.add_argument("--no_amp", action='store_true')
    parser.add_argument("--no_cuda", action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    # Set seed for reproducibility
    set_seed(3407)

    # Parse arguments
    args = parse_args()

    # Ensure output directory exists
    out_path = f'{args.root}/model/{args.save_model_name}'
    os.makedirs(out_path, exist_ok=True)

    # 直接调用main函数，不使用multiprocessing.spawn
    main(args)
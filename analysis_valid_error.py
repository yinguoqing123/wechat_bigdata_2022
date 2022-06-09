import logging
import os
import time
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from functools import partial
from sklearn.model_selection import train_test_split
import pandas as pd
import json

from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal
from wx_uni_model import WXUniModel
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from create_optimizer import create_optimizer, get_reducelr_schedule, get_warmup_schedule
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id, category_id_to_lv2id

def create_dataloaders(args):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
    size = len(dataset)
    train_indices, test_indices = train_test_split(list(range(len(dataset.labels))), test_size=args.val_ratio, random_state=2022, 
                                                    stratify=dataset.labels)
    train_dataset, val_dataset = torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, test_indices)
    
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)
        
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return dataset, val_dataset, val_dataloader


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)
    
    model.train()
    return loss, results


def get_valid_predict_label(args):
    # 1. load data
    dataset, val_dataset, val_dataloader = create_dataloaders(args)
    
    # 2. load model
    model = WXUniModel(task=[], model_path=args.bert_dir)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
        
    model.eval()
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
        # model = model.to(args.device)
        
    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in val_dataloader:
            _, _, pred_label_id, label_id = model(batch)
            predictions.extend(pred_label_id.cpu().numpy())
        
    # 4. dump results
    val_anns = [dataset.anns[idx] for idx in val_dataset.indices]
    for i in range(len(val_anns)):
        val_anns[i]['predict_id'] = lv2id_to_category_id(predictions[i])

    df = pd.DataFrame(val_anns)
    df.to_csv("valid_predict.csv", index=False)
                    

if __name__ == '__main__':
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)
    get_valid_predict_label(args)

    
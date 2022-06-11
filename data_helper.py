import json
import random
import zipfile
from io import BytesIO
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, AutoTokenizer
from sklearn.model_selection import train_test_split
from collections import Counter
import re

from category_id_map import category_id_to_lv2id, category_id_to_lv1id


def create_dataloaders(args, pretrain=False):
    if pretrain:
        dataset = MultiModalDataset(args, args.pretrain_annotation, args.pretrain_zip_feats, test_mode=True)
        size = len(dataset)
        val_size = 10000
        train_dataset, val_dataset = torch.utils.data.dataset.random_split(dataset, [size - val_size, val_size])
    else:
        dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
        size = len(dataset)
        val_size = int(size * args.val_ratio)
        # 分层抽样
        train_indices, test_indices = train_test_split(list(range(len(dataset.labels))), test_size=args.val_ratio, random_state=2022, 
                                                        stratify=dataset.labels)
        train_dataset, val_dataset = torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, test_indices)
        resample(train_dataset)
        
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
    return train_dataloader, val_dataloader


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.
    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.
    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """
    
    def __init__(self,
                 args,
                 ann_path: str,
                 zip_feats: str,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode
        self.zip_feat_path = zip_feats
        self.num_workers = args.num_workers
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        # initialize the text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache, do_lower_case=True)
        self.labels = None
        if not test_mode:
            self.labels = [self.anns[idx]['category_id'] for idx in range(len(self.anns))]
        
    def __len__(self) -> int:
        return len(self.anns)
    
    def get_visual_feats(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape
        
        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask
    
    def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask
    
    def tokenize_text2(self, title: str, ocr_text: str, asr_text: str) -> tuple:
        encoded_titles = self.tokenizer(title, max_length=80, truncation=True, add_special_tokens=False)
        encoded_ocr = self.tokenizer(ocr_text, max_length=200, truncation=True, add_special_tokens=False)
        encoded_asr = self.tokenizer(asr_text, max_length=200, truncation=True, add_special_tokens=False)
        text_input_ids = [self.tokenizer.cls_token_id] + encoded_titles['input_ids'][:80] + [self.tokenizer.sep_token_id] + \
            encoded_ocr['input_ids'][:128] + [self.tokenizer.sep_token_id] + encoded_asr['input_ids'][:128] + [self.tokenizer.sep_token_id]
        text_input_ids = torch.LongTensor(text_input_ids[:340] + [self.tokenizer.pad_token_id] * (340 - len(text_input_ids)))
        text_mask = [1,] + encoded_titles['attention_mask'][1:-1] + [1,] + encoded_ocr['attention_mask'][1:-1] + [1, ] + \
            encoded_asr['attention_mask'][1:-1] + [1, ]
        text_mask = torch.LongTensor(text_mask[:340] + [0] * (340 - len(text_mask)))
        text_token_type_ids = torch.zeros_like(text_input_ids)
        return text_input_ids, text_mask, text_token_type_ids
    
    def tokenize_img(self, idx: int) -> tuple:
        frame_input, frame_mask = self.get_visual_feats(idx)
        frame_token_type_ids = torch.ones_like(frame_mask)
        return frame_input, frame_mask, frame_token_type_ids
    
    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        # frame_input, frame_mask = self.get_visual_feats(idx)
        
        # Step 2, load title tokens
        # title_input, title_mask = self.tokenize_text(self.anns[idx]['title'])
        
        
        # title ocr asr
        title, asr = self.anns[idx]['title'], self.anns[idx]['asr']
        asr = re.sub('嗯{3,}', '', asr)
        if len(asr) > 128:
            asr = asr[:64] + ',' + asr[-64:]
        ocr = sorted(self.anns[idx]['ocr'], key = lambda x: x['time'])
        ocr = ','.join([t['text'] for t in ocr])
        if len(ocr) > 128:
            ocr = ocr[:64] + ',' + ocr[-64:]
        text_input, text_mask, text_token_type_ids = self.tokenize_text2(title, ocr, asr)
        frame_input, frame_mask, frame_token_type_ids = self.tokenize_img(idx)
        
        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            frame_token_type_ids=frame_token_type_ids,
            text_input=text_input,
            text_mask=text_mask,
            text_token_type_ids=text_token_type_ids
        )
        
        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            lebel_lv1 = category_id_to_lv1id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])
            data['label_lv1'] = torch.LongTensor([lebel_lv1])
            
        return data
    
    
# 计算lv1 lv2 的先验概率
def get_prior_lv1_lv2(ann_path='../data/annotations/labeled.json'):
    with open(ann_path, 'r', encoding='utf8') as f:
        anns = json.load(f)
        
    lv2 = [line['category_id'] for line in anns]
    lv2 = Counter(lv2)
    cnt_lv2 = sum(lv2.values())
    lv2_prior = {}
    for key in lv2:
        lv2_prior[category_id_to_lv2id(key)] = lv2[key] / cnt_lv2
            
    lv2_prior = sorted(lv2_prior.items(), key=lambda x: x[0])
    lv2_prior = np.array([np.log(x[1]) for x in lv2_prior])
            
    lv1 = [line['category_id'][:2] for line in anns]
    lv1 = Counter(lv1)
    cnt_lv1 = sum(lv1.values())
    lv1_prior = {}
    for key in lv1:
        lv1_prior[key] = lv1[key] / cnt_lv1
            
    lv1_prior = sorted(lv1_prior.items(), key=lambda x: x[0])
    lv1_prior = np.array([np.log(x[1]) for x in lv1_prior])
    return lv1_prior, lv2_prior

def resample(dataset): 
    anns = dataset.dataset.anns
    indices = dataset.indices
    labels = [line['category_id'] for line in anns]
    label_cnt = Counter(labels)
    indices_resample = []
    for idx in indices:
        if label_cnt[anns[idx]['category_id']] < 100:
            indices_resample.extend([idx] * 5)
        elif label_cnt[anns[idx]['category_id']] < 500:
            indices_resample.extend([idx] * 3)
        elif label_cnt[anns[idx]['category_id']] < 1000:
            indices_resample.extend([idx] * 2)
        else:
            indices_resample.append(idx)
    
    dataset.indices = indices_resample
    print("trainset len:", len(dataset))
    
def get_weight_lv1_lv2(ann_path='../data/annotations/labeled.json'):
    with open(ann_path, 'r', encoding='utf8') as f:
        anns = json.load(f)
    
    lv1 = [int(line['category_id'][:2]) for line in anns]
    lv1 = Counter(lv1)
    lv1 = sorted(lv1.items(), key=lambda x: x[0])
    weight_lv1 = []
    for id, cnt in lv1:
        if cnt > 5000:
            weight_lv1.append(0.5)
        else:
            weight_lv1.append(1)
            
    lv2 = [line['category_id'] for line in anns]
    lv2 = Counter(lv2)
    lv2 = sorted(lv2.items(), key=lambda x: x[0])
    weight_lv2 = []
    for id, cnt in lv2:
        if cnt > 3000:
            weight_lv2.append(0.5)
        else:
            weight_lv2.append(1)
    return weight_lv1, weight_lv2



import logging
import random
import math

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup

from category_id_map import lv2id_to_lv1id


def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def build_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    return optimizer, scheduler


def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'mean_f1': mean_f1}

    return eval_results

class ArcFace(nn.Module):
    """ NN module for projecting extracted embeddings onto the sphere surface """
    
    def __init__(self, in_features, out_features, s=30, m=0.5):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.arc_min = math.cos(math.pi - self.m)
        self.margin_min = math.sin(math.pi - self.m) * self.m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def _update_margin(self, new_margin):
        self.m = new_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.arc_min = math.cos(math.pi - self.m)
        self.margin_min = math.sin(math.pi - self.m) * self.m
        
    def forward(self, embedding, label):
            cos = F.linear(F.normalize(embedding), F.normalize(self.weight))
            sin = torch.sqrt(1.0 - torch.pow(cos, 2)).clamp(0, 1)
            phi = cos * self.cos_m - sin * self.sin_m
            phi = torch.where(cos > self.arc_min, phi, cos - self.margin_min)

            one_hot = torch.zeros(cos.size(), device=embedding.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            logits = one_hot * phi + (1.0 - one_hot) * cos
            logits *= self.s
            return logits
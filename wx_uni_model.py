#%%writefile qqmodel/qq_uni_model.py
import imp
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
import copy

import sys
# sys.path.append("..")
from masklm import MaskLM, MaskVideo, ShuffleVideo
from transformers import AutoTokenizer
from modeling_bert import *
from data_helper import get_prior_lv1_lv2
from util import ArcFace, FocalLoss 

class WXUniModel(nn.Module):
    def __init__(self, args, task=[], init_from_pretrain=True, use_arcface_loss=False):
        super().__init__()
        uni_bert_cfg = BertConfig.from_pretrained(args.bert_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_dir)
        # self.frame_fc = nn.Linear(768, uni_bert_cfg.hidden_size, bias=False)
        
        #uni_bert_cfg.num_hidden_layers = 1
        
        self.use_arcface_loss = use_arcface_loss

        self.text_classify = Classify(768, name='text', use_arcface_loss=use_arcface_loss)
        self.frame_classify = Classify(768, name='frame', use_arcface_loss=use_arcface_loss)
        # self.union_classify = Classify(768*2, name='union', use_arcface_loss=use_arcface_loss)
        self.mid_text_classify = Classify(768, name='text_mid', use_arcface_loss=use_arcface_loss)
        self.mid_frame_classify = Classify(768, name='text_mid', use_arcface_loss=use_arcface_loss)

        if init_from_pretrain:
            self.roberta = UniBertForMaskedLM.from_pretrained(args.bert_dir, config=uni_bert_cfg)
        else:
            self.roberta = UniBertForMaskedLM(uni_bert_cfg)
            
        self.roberta.bert.set_cls_token_id(int(self.tokenizer.cls_token_id))

    def forward(self, inputs, target=None, task=None, inference=False, pretrain=False):
        loss, pred = 0, None
        frame_feature, frame_mask, frame_token_type_ids = inputs['frame_input'], inputs['frame_mask'], inputs['frame_token_type_ids']
        text_input_ids, text_mask, text_token_type_ids = inputs['text_input'], inputs['text_mask'], inputs['text_token_type_ids']
            
        union_mask = torch.cat([text_mask, frame_mask], dim=-1)
        text_mask, frame_mask, union_mask = text_mask.float(), frame_mask.float(), union_mask.float()
        output, _ = self.roberta(text_input_ids, text_mask, text_token_type_ids, frame_token_type_ids=frame_token_type_ids, 
                                    frame_mask=frame_mask, inputs_embeds=frame_feature, modal_type='union', 
                                    output_hidden_states=False)
        text_output, frame_output, union_output = output['text_embeddings'], output['frame_embeddings'], output['union_embeddings']
        
        text_mid_output, frame_mid_output = output['mid_frame_embeddings'], output['mid_text_embeddings']
        # text_pooling = torch.sum(text_output * text_mask.unsqueeze(dim=-1), dim=1) / torch.sum(text_mask, dim=-1, keepdim=True)
        # frame_pooling = torch.sum(frame_output * frame_mask.unsqueeze(dim=-1), dim=1) / torch.sum(frame_mask, dim=-1, keepdim=True)
        # union_pooling = torch.sum(union_output * union_mask.unsqueeze(dim=-1), dim=1) / torch.sum(union_mask, dim=-1, keepdim=True)
        text_pooling = text_output[:, 0, :]
        frame_pooling = frame_output[:, 0, :]
        text_mid_pooling = text_mid_output[:, 0, :]
        frame_mid_pooling = frame_mid_output[:, 0, :]
        union_pooling = torch.cat([text_pooling, frame_pooling], dim=-1)
        
        if self.use_arcface_loss:
            text_logits1, text_logits2 = self.text_classify(text_pooling, inputs['label_lv1'], inputs['label'])  
            frame_logits1, frame_logits2 = self.frame_classify(frame_pooling, inputs['label_lv1'], inputs['label'])  
            union_logits1, union_logits2 = self.union_classify(union_pooling, inputs['label_lv1'], inputs['label'])
        else:
            text_logits1, text_logits2 = self.text_classify(text_pooling)  
            frame_logits1, frame_logits2 = self.frame_classify(frame_pooling)  
            # union_logits1, union_logits2 = self.union_classify(union_pooling)
            union_logits1, union_logits2 = (text_logits1+frame_logits1)/2, (text_logits2+frame_logits2)/2
            mid_text_logits1, mid_text_logits2 = self.mid_text_classify(text_mid_pooling)
            mid_frame_logits1, mid_frame_logits2 = self.mid_frame_classify(frame_mid_pooling)
        
         
        if inference:
            return torch.argmax(union_logits2, dim=1)
        else:
            text_result = self.cal_loss(text_logits1, text_logits2, inputs['label_lv1'], inputs['label'], focal_loss=False)
            frame_result = self.cal_loss(frame_logits1, frame_logits2, inputs['label_lv1'], inputs['label'], focal_loss=False)
            union_result = self.cal_loss(union_logits1, union_logits2, inputs['label_lv1'], inputs['label'], focal_loss=False)
            mid_text_result = self.cal_loss(mid_text_logits1, mid_text_logits2, inputs['label_lv1'], inputs['label'], focal_loss=False)
            mid_frame_result = self.cal_loss(mid_frame_logits1, mid_frame_logits2, inputs['label_lv1'], inputs['label'], focal_loss=False)
            loss = 0.5 * text_result['loss'] + 0.5 * frame_result['loss'] + union_result['loss'] + 0.25 * mid_text_result['loss'] + 0.25 * mid_frame_result['loss']
            return  loss, text_result, frame_result, union_result # loss_category, accuracy, pred_label_id, label 
    
    @staticmethod
    def cal_loss(prediction_lv1, prediction_lv2, label_lv1, label_lv2, focal_loss=False):
        weight_lv1 = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 1, 1, 1, 0.5, 0.5, 0.5, 1, 1, 0.5], device='cuda')
        weight_lv2 = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 0.5, 1, 1, 0.5, 1, 1, 1, 1, 1, 0.5, 1, 1, 1, 1, 1, 1, 1, 0.5, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 0.5, 1, 1, 1, 1, 0.5, 1, 1, 1, 1, 1, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda')
        label_lv1 = label_lv1.squeeze(dim=1)
        label_lv2 = label_lv2.squeeze(dim=1)
        if focal_loss:
            loss_lv1 = FocalLoss()(prediction_lv1, label_lv1)
            loss_lv2 = FocalLoss()(prediction_lv2, label_lv2)
        else:
            loss_lv1 = F.cross_entropy(prediction_lv1, label_lv1, weight_lv1)
            loss_lv2 = F.cross_entropy(prediction_lv2, label_lv2, weight_lv2)
        loss = loss_lv1 + loss_lv2
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction_lv2, dim=1)
            accuracy = (label_lv2 == pred_label_id).float().sum() / label_lv2.shape[0]
        result = {'loss': loss, 'accuracy': accuracy, 'pred_label_id': pred_label_id, 'label_lv2': label_lv2}
        return result


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

    
class UniBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config, cls_head=False):
        super().__init__(config)
        self.bert = UniBert(config)
        if cls_head:
            self.cls = BertOnlyMLMHead(config)
        
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, input_ids=None, mask=None, token_type_ids=None, gather_index=None, inputs_embeds=None, modal_type='text', frame_mask=None, 
                frame_token_type_ids=None, output_hidden_states=False, return_mlm=False):
        encoder_outputs = self.bert(input_ids, mask, token_type_ids, gather_index, inputs_embeds, modal_type, frame_mask,
                                    frame_token_type_ids, output_hidden_states)
        frame_len = frame_mask.size()[1]
        if return_mlm:
            return encoder_outputs, self.cls(encoder_outputs)[:, :-frame_len, :]
        else:
            return encoder_outputs, None        
        
class UniBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.frame_fc = nn.Linear(768, self.config.hidden_size, bias=False)
        self.encoder = BertEncoder(config)
        # frame_config = copy.deepcopy(config)
        # frame_config.num_hidden_layers = 3
        # self.frame_encoder = BertEncoder(frame_config)
        self.cls_token_id = None
        self.unused_token_id = [1, 2, 3]

        self.init_weights()
        
    def set_cls_token_id(self, cls_token_id):
        self.cls_token_id = cls_token_id

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, input_ids=None, mask=None, token_type_ids=None, gather_index=None, inputs_embeds=None, modal_type='text', frame_mask=None, 
                frame_token_type_ids=None, output_hidden_states=True):

        # 前6层各自模态单独编码
        inputs_embeds = self.frame_fc(inputs_embeds)
        frame_embeddings = self.embeddings(inputs_embeds=inputs_embeds, token_type_ids=frame_token_type_ids)
        cls_embeddings = self.embeddings.word_embeddings(torch.tensor([self.cls_token_id], device=mask.device)).expand(frame_mask.shape[0], 1, -1)
        frame_mask = torch.cat([torch.ones_like(frame_mask[:, :1]), frame_mask], dim=-1)
        frame_embeddings = torch.cat([cls_embeddings, frame_embeddings], dim=1)
        
        text_embeddings = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        
        single_embeddings = torch.cat([text_embeddings, frame_embeddings], dim=1)
        single_text_mask = (torch.cat([mask, torch.zeros_like(frame_mask)], dim=-1)[:, None, None, :]).expand(-1, -1, mask.shape[1], -1)
        single_text_mask  = ((1.0 - single_text_mask ) * -1000000.0).float()
        single_frame_mask = (torch.cat([torch.zeros_like(mask), frame_mask,], dim=-1)[:, None, None, :]).expand(-1, -1, frame_mask.shape[1], -1)
        single_frame_mask  = ((1.0 - single_frame_mask ) * -1000000.0).float()
        single_mask = torch.cat([single_text_mask, single_frame_mask], dim=2)
        
        single_output_embeddings = self.encoder(single_embeddings, single_mask, start_layer=0, end_layer=4)['last_hidden_state']
        single_text_mid_output_embeddings, single_frame_mid_output_embeddings = single_output_embeddings[:, :single_text_mask.shape[2], :], single_output_embeddings[:, single_text_mask.shape[2]:, :]
        
        # 后六层 添加共享的unused_token
        unused_embeddings = self.embeddings.word_embeddings(torch.tensor(self.unused_token_id, device=mask.device)).expand(frame_mask.shape[0], -1, -1)
        union_input_embeddings = torch.cat([single_text_mid_output_embeddings, unused_embeddings, unused_embeddings, single_frame_mid_output_embeddings], dim=1)
        union_text_mask = (torch.cat([mask, torch.ones(mask.shape[0], len(self.unused_token_id), device=mask.device), torch.zeros(mask.shape[0], len(self.unused_token_id), device=mask.device), torch.zeros_like(frame_mask)], dim=-1)[:, None, None, :]).expand(-1, -1, mask.shape[1], -1)
        union_text_mask  = ((1.0 - union_text_mask ) * -1000000.0).float()
        union_frame_mask = (torch.cat([torch.zeros_like(mask), torch.zeros(frame_mask.shape[0], len(self.unused_token_id), device=frame_mask.device), torch.ones(frame_mask.shape[0], len(self.unused_token_id), device=frame_mask.device), frame_mask], dim=-1)[:, None, None, :]).expand(-1, -1, frame_mask.shape[1], -1)
        union_frame_mask  = ((1.0 - union_frame_mask ) * -1000000.0).float()
        unused_mask1 = (torch.cat([torch.zeros_like(mask), torch.ones(mask.shape[0], len(self.unused_token_id), device=mask.device), torch.zeros(mask.shape[0], len(self.unused_token_id), device=mask.device), frame_mask], dim=-1)[:, None, None, :]).expand(-1, -1, len(self.unused_token_id), -1)
        unused_mask1  = ((1.0 - unused_mask1 ) * -1000000.0).float()
        unused_mask2 = (torch.cat([mask, torch.zeros(mask.shape[0], len(self.unused_token_id), device=mask.device), torch.ones(mask.shape[0], len(self.unused_token_id), device=mask.device), torch.zeros_like(frame_mask)], dim=-1)[:, None, None, :]).expand(-1, -1, len(self.unused_token_id), -1)
        unused_mask2  = ((1.0 - unused_mask2 ) * -1000000.0).float()
        union_mask = torch.cat([union_text_mask , unused_mask1, unused_mask2, union_frame_mask], dim=2)
        union_output_embeddings =  self.encoder(union_input_embeddings, union_mask, start_layer=4, end_layer=12)['last_hidden_state']

        output_embeddings_text, output_embeddings_frame = union_output_embeddings[:, :single_text_mask.shape[2], :], union_output_embeddings[:, -single_frame_mask.shape[2]:, :]
        
        output = dict(
            frame_embeddings=output_embeddings_frame, 
            text_embeddings=output_embeddings_text,
            union_embeddings=union_output_embeddings,
            mid_frame_embeddings = single_frame_mid_output_embeddings,
            mid_text_embeddings = single_text_mid_output_embeddings
        )
        return output
    
    def get_extended_mask(self, mask):
        mask = mask[:, None, None, :]
        mask = ((1.0 - mask) * -1000000.0).float()
        return mask
    
class Classify(nn.Module):
    def __init__(self, in_features, name='', use_arcface_loss=False) -> None:
        super().__init__()
        self.name = name
        self.use_arcface_loss=use_arcface_loss
        if use_arcface_loss:
            self.fc1 = ArcFace(in_features, 23, s=20, m=0.2)
            self.fc2 = ArcFace(in_features, 200, s=20, m=0.1)
        else:
            self.fc1 = torch.nn.Linear(in_features, 23)
            self.fc2 = torch.nn.Linear(in_features, 200)
            
        prior_lv1, prior_lv2 = get_prior_lv1_lv2()
        prior_lv1, prior_lv2 = torch.tensor(prior_lv1, device='cuda', dtype=torch.float), torch.tensor(prior_lv2, device='cuda', dtype=torch.float)
        self.register_buffer('prior_lv1', prior_lv1)
        self.register_buffer('prior_lv2', prior_lv2)
        
    def forward(self, input, label1=None, label2=None):
        if self.use_arcface_loss:
            logits1 = self.fc1(input, label1) + self.prior_lv1.unsqueeze(dim=0)
            logits2 = self.fc2(input, label2) + self.prior_lv2.unsqueeze(dim=0)
        else:
            logits1 = self.fc1(input) + self.prior_lv1.unsqueeze(dim=0)
            logits2 = self.fc2(input) + self.prior_lv2.unsqueeze(dim=0)
        return logits1, logits2

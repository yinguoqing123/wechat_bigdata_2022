import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING, BertModel, BertConfig

from category_id_map import CATEGORY_ID_LIST
from modeling_bert import BertEncoder
from util import ArcFace, FocalLoss
from data_helper import get_prior_lv1_lv2


class WXUniModel(nn.Module):
    def __init__(self, args, task=[], init_from_pretrain=True, use_arcface_loss=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        config = BertConfig.from_pretrained(args.bert_dir)
        config.num_hidden_layers = 3
        self.frame_encoder = BertEncoder(config)
        self.fusion_encoder = BertEncoder(config)
    
        self.classifier_text = Classify(name='text')
        self.classifier_frame = Classify(name='frame')
        self.classifier_fusion = Classify(name='fustion')

    def forward(self, inputs, inference=False):
        frame_feature, frame_mask, frame_token_type_ids = inputs['frame_input'], inputs['frame_mask'], inputs['frame_token_type_ids']
        text_input_ids, text_mask, text_token_type_ids = inputs['text_input'], inputs['text_mask'], inputs['text_token_type_ids']
          
        bert_embedding = self.bert(text_input_ids, text_mask)['last_hidden_state']
        
        frame_feature = self.bert.embeddings(inputs_embeds=frame_feature, token_type_ids=frame_token_type_ids)
        frame_mask_extended= self.get_extended_attention_mask(frame_mask)
        frame_embedding = self.frame_encoder(frame_feature, frame_mask_extended)['last_hidden_state']
        
        fusion_feature = torch.cat([bert_embedding, frame_embedding], dim=1)
        fusion_mask = torch.cat([text_mask, frame_mask], dim=-1)
        fusion_mask_extende = self.get_extended_attention_mask(fusion_mask)
        fusion_embedding = self.fusion_encoder(fusion_feature, fusion_mask_extende)['last_hidden_state']
        
        logit_lv1_text, logit_lv2_text = self.classifier_text(torch.mean(bert_embedding, dim=1))
        logit_lv1_frame, logit_lv2_frame = self.classifier_frame(torch.mean(frame_embedding, dim=1))
        logit_lv1_fusion, logit_lv2_fusion = self.classifier_fusion(torch.mean(fusion_embedding, dim=1))

        if inference:
            return torch.argmax(logit_lv2_fusion, dim=1)
        else:
            text_result = self.cal_loss(logit_lv1_text, logit_lv2_text, inputs['label_lv1'], inputs['label'], focal_loss=False)
            frame_result = self.cal_loss(logit_lv1_frame, logit_lv2_frame, inputs['label_lv1'], inputs['label'], focal_loss=False)
            union_result = self.cal_loss(logit_lv1_fusion, logit_lv2_fusion, inputs['label_lv1'], inputs['label'], focal_loss=False)
            loss = 0.5 * text_result['loss'] + 0.5 * frame_result['loss'] + union_result['loss'] 
            return loss, text_result, frame_result, union_result
        
    def get_extended_attention_mask(self, mask):
        mask = mask[:, None, None, :]   # [batch_size, head_num, from_seq_length, to_seq_length]
        mask = ((1.0 - mask) * -1000000.0).float()
        return mask

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


class Classify(nn.Module):
    def __init__(self, name='', use_arcface=False) -> None:
        super().__init__()
        self.name = name
        self.use_arcface = use_arcface
        if use_arcface:
            self.fc1 = ArcFace(768, 23, s=20, m=0.2)
            self.fc2 = ArcFace(768, 200, s=20, m=0.05)
        else:
            self.fc1 = nn.Linear(768, 23)
            self.fc2 = nn.Linear(768, 200)
            
        prior_lv1, prior_lv2 = get_prior_lv1_lv2()
        prior_lv1, prior_lv2 = torch.tensor(prior_lv1, device='cuda', dtype=torch.float), torch.tensor(prior_lv2, device='cuda', dtype=torch.float)
        self.register_buffer('prior_lv1', prior_lv1)
        self.register_buffer('prior_lv2', prior_lv2)
        
    def forward(self, input, label=None):
        if self.use_arcface:
            logits1 = self.fc1(input, label) + self.prior_lv1.unsqueeze(dim=0)
            logits2 = self.fc2(input, label) + self.prior_lv2.unsqueeze(dim=0)
        else:
            logits1 = self.fc1(input) + self.prior_lv1.unsqueeze(dim=0)
            logits2 = self.fc2(input) + self.prior_lv2.unsqueeze(dim=0)
        return logits1, logits2
 
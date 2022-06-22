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

class WXUniPretrainModel(nn.Module):
    def __init__(self, args, task=['mlm', 'itm'], init_from_pretrain=True, use_arcface_loss=False):
        super().__init__()
        uni_bert_cfg = BertConfig.from_pretrained(args.bert_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_dir)
        self.config = uni_bert_cfg
        
        #uni_bert_cfg.num_hidden_layers = 1
        
        self.use_arcface_loss = use_arcface_loss
    
        self.task = set(task)
        
        if 'mlm' in task:
            self.lm = MaskLM(tokenizer_path=args.bert_dir)
        
        if 'mvm' in task:
            self.vm = MaskVideo()
            self.roberta_mvm_lm_header = VisualOnlyMLMHead(uni_bert_cfg) 
            
        if 'itm' in task:
            self.sv = ShuffleVideo()
            self.newfc_itm_text_cls = torch.nn.Linear(uni_bert_cfg.hidden_size, 1) 
            self.newfc_itm_frame_cls = torch.nn.Linear(uni_bert_cfg.hidden_size, 1) 

        if init_from_pretrain:
            self.roberta = UniBertForMaskedLM.from_pretrained(args.bert_dir, config=uni_bert_cfg)
        else:
            self.roberta = UniBertForMaskedLM(uni_bert_cfg)
            
        self.roberta.bert.set_cls_token_id(int(self.tokenizer.cls_token_id))

    def forward(self, inputs, target=None, task=None, inference=False, pretrain=False):
        loss, pred = 0, None
        frame_feature, frame_mask, frame_token_type_ids = inputs['frame_input'], inputs['frame_mask'], inputs['frame_token_type_ids']
        text_input_ids, text_mask, text_token_type_ids = inputs['text_input'], inputs['text_mask'], inputs['text_token_type_ids']
        
        if task is None:
            sample_task = self.task
        elif type(task) == str:
            sample_task = [task]
        elif type(task) == list:
            sample_task = task
        
        # # perprocess
        return_mlm = False
        if 'mlm' in sample_task:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids = input_ids.to(text_input_ids.device)
            lm_label = lm_label.to(text_input_ids.device)
            return_mlm = True
            
        if 'itm' in sample_task:
            input_feature, video_text_match_label = self.sv.torch_shuf_video(frame_feature.cpu())
            frame_feature = input_feature.to(frame_feature.device)
            video_text_match_label = video_text_match_label.to(frame_feature.device)
            
        if 'mvm' in sample_task:
            vm_input = frame_feature
            input_feature, video_label = self.vm.torch_mask_frames(frame_feature.cpu(), frame_mask.cpu())
            frame_feature = input_feature.to(frame_feature.device)
            video_label = video_label.to(frame_feature.device)
        
        # 
        union_mask = torch.cat([text_mask, frame_mask], dim=-1)
        text_mask, frame_mask, union_mask = text_mask.float(), frame_mask.float(), union_mask.float()
        output_embeddings, lm_prediction_scores = self.roberta(text_input_ids, text_mask, text_token_type_ids, frame_token_type_ids=frame_token_type_ids, 
                                    frame_mask=frame_mask, inputs_embeds=frame_feature, modal_type='union', 
                                    output_hidden_states=False, return_mlm=return_mlm)
        
        # 带cls拼接的
        output_frame_embeddings, output_text_embeddings = output_embeddings['frame_embeddings'], output_embeddings['text_embeddings']
        text_embeddings_mid,  frame_embeddings_mid = output_embeddings['mid_text_embeddings'], output_embeddings['mid_frame_embeddings']
        
        # 对比损失
        if 'itc' in sample_task:
            # 只取一半， 另一半当itm任务存在时，是打乱的 
            text_embeddings_mid_norm, frame_embeddings_mid_norm = F.normalize(text_embeddings_mid[:, 0, :], dim=-1), F.normalize(frame_embeddings_mid[:, 0, :], dim=-1),
            bs = text_mask.shape[0] // 2
            sim_text_frame = torch.matmul(text_embeddings_mid_norm[:bs], frame_embeddings_mid_norm[:bs].t()) * 10  # (bsz, bsz)
            labels = torch.eye(bs)
            itc_loss = (torch.log_softmax(sim_text_frame, dim=-1) * labels).sum() / bs
        
        if 'mlm' in sample_task:
            mlm_pred = lm_prediction_scores.contiguous().view(-1, self.config.vocab_size)
            mlm_loss = nn.CrossEntropyLoss()(mlm_pred, lm_label.view(-1))
            mlm_accuracy = torch.sum(lm_prediction_scores.argmax(dim=-1).view(-1)==lm_label.view(-1)) / (torch.sum(lm_label.view(-1)>0) + 1e-12)
            # mlm_loss = torch.log(mlm_loss + 1e-12)
            
        if 'mvm' in sample_task:
            vm_output = self.roberta_mvm_lm_header(output_frame_embeddings[:, 1:, :])
            masked_vm_loss = self.calculate_mfm_loss(vm_output, vm_input, 
                                                     frame_mask, video_label, normalize=False)
        else:
            masked_vm_loss = torch.tensor(0, device='cuda')
            
        if 'itm' in sample_task:
            itm_pred_text_cls = self.newfc_itm_text_cls(output_text_embeddings[:, 0, :])
            itm_pred_frame_cls = self.newfc_itm_frame_cls(output_frame_embeddings[:, 0, :])
            itm_pred = torch.cat([itm_pred_text_cls.view(-1), itm_pred_frame_cls.view(-1)])
            video_text_match_label = torch.cat([video_text_match_label.view(-1), video_text_match_label.view(-1)])
            loss_itm = nn.BCEWithLogitsLoss()(itm_pred, video_text_match_label)
            itm_accuracy = torch.sum( (itm_pred>0.5).int() == video_text_match_label.int() ).float() / itm_pred.view(-1).shape[0]
            # loss_itm += torch.log(loss_itm + 1e-12)
         
        return   mlm_loss, loss_itm, mlm_accuracy, itm_accuracy, masked_vm_loss, itc_loss
    
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
        
    
    def calculate_mfm_loss(self, frame_feature_output, frame_feature_input, 
                           frame_mask, video_labels_index, normalize=False, temp=0.1):
        if normalize:
            frame_feature_output = torch.nn.functional.normalize(frame_feature_output, p=2, dim=2)
            frame_feature_input = torch.nn.functional.normalize(frame_feature_input, p=2, dim=2)

        afm_scores_tr = frame_feature_output.view(-1, frame_feature_output.shape[-1])

        video_tr = frame_feature_input.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        if normalize:
            logits_matrix = logits_matrix / temp

        frame_mask_float = frame_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(frame_mask_float.view(-1, 1), frame_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != -100)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss

def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, 768, bias=False)
        self.bias = nn.Parameter(torch.zeros(768))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
    
class UniBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = UniBert(config)
        self.cls = BertOnlyMLMHead(config)
        
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, input_ids=None, mask=None, token_type_ids=None, gather_index=None, inputs_embeds=None, modal_type='text', frame_mask=None, 
                frame_token_type_ids=None, output_hidden_states=True, return_mlm=False):
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
        # self.video_fc1 = torch.nn.Linear(768, config.hidden_size)
        # self.video_fc2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        # self.video_embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # frame_config = copy.deepcopy(config)
        # frame_config.num_hidden_layers = 3
        # self.frame_encoder = BertEncoder(frame_config)
        self.cls_token_id = None
        self.frame_fc = nn.Linear(768, self.config.hidden_size, bias=False)

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
        
        single_output_embeddings = self.encoder(single_embeddings, single_mask, start_layer=0, end_layer=6)['last_hidden_state']
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
        union_output_embeddings =  self.encoder(union_input_embeddings, union_mask, start_layer=6, end_layer=12)['last_hidden_state']

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


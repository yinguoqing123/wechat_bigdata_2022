#%%writefile qqmodel/qq_uni_model.py
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
# sys.path.append("..")
from masklm import MaskLM, MaskVideo, ShuffleVideo
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder

class WXUniModel(nn.Module):
    def __init__(self, cfg, bert_cfg_dict, model_path, task=['mlm', 'mvm', 'itm'], init_from_pretrain=True):
        super().__init__()
        uni_bert_cfg = BertConfig.from_pretrained(f'{model_path}/config.json')
        #uni_bert_cfg.num_hidden_layers = 1
        
        self.frame_dense = torch.nn.Linear(768, 768)
        self.frame_dropout = torch.nn.Dropout(0.1)
        self.classify_dense = torch.nn.Linear(768, 200)
        self.ln = torch.nn.LayerNorm(eps=uni_bert_cfg.layer_norm_eps)
        
        self.task = set(task)
        
        if 'mlm' in task:
            self.lm = MaskLM(tokenizer_path=model_path)
        
        if 'mvm' in task:
            self.vm = MaskVideo()
            self.roberta_mvm_lm_header = VisualOnlyMLMHead(uni_bert_cfg) 
            
        if 'itm' in task:
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(uni_bert_cfg.hidden_size, 1) 

        if init_from_pretrain:
            self.roberta = UniBertForMaskedLM.from_pretrained(model_path, config=uni_bert_cfg)
        else:
            self.roberta = UniBertForMaskedLM(uni_bert_cfg)

    def forward(self, inputs, target=None, task=None, inference=False):
        loss, pred = 0, None
        frame_feature, frame_mask, frame_token_type_ids = inputs['frame_feature'], inputs['frame_mask'], inputs['frame_token_type_ids']
        text_input_ids, text_mask, text_token_type_ids = inputs['text_input_ids'], inputs['text_mask'], inputs['text_token_type_ids']
        
        # frame feature 映射到同一空间
        frame_feature = self.frame_dense(frame_feature)
        frame_feature = self.frame_dropout(self.ln(frame_feature))
        
        if task is None:
            sample_task = self.task
        elif type(task) == str:
            sample_task = [task]
        elif type(task) == list:
            sample_task = task
        
        # perprocess
        return_mlm = False
        if 'mlm' in sample_task:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids = input_ids.to(text_input_ids.device)
            lm_label = lm_label.to(text_input_ids.device)
            return_mlm = True
            
        if 'mvm' in sample_task:
            vm_input = frame_feature
            input_feature, video_label = self.vm.torch_mask_frames(frame_feature.cpu(), frame_mask.cpu())
            frame_feature = input_feature.to(frame_feature.device)
            video_label = video_label.to(frame_feature.device)
            
        if 'itm' in sample_task:
            input_feature, video_text_match_label = self.sv.torch_shuf_video(frame_feature.cpu())
            frame_feature = input_feature.to(frame_feature.device)
            video_text_match_label = video_text_match_label.to(frame_feature.device)
            
        # concat features
        features, lm_prediction_scores = self.roberta(inputs, return_mlm=return_mlm)
        prediction = self.classify_dense(features[:, 0, :])
        
        # compute loss
        loss_category, accuracy, pred_label_id, label = self.cal_loss(prediction, inputs['label'])
        
        if 'mlm' in sample_task:
            pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
            masked_lm_loss = nn.CrossEntropyLoss()(pred, lm_label.view(-1))
            loss += masked_lm_loss / 1.25 / len(sample_task)
            
        if 'mvm' in sample_task:
            vm_output = self.roberta_mvm_lm_header(features[:, text_input_ids.size()[1]:, :])
            masked_vm_loss = self.calculate_mfm_loss(vm_output, vm_input, 
                                                     frame_mask, video_label, normalize=False)
            loss += masked_vm_loss / 3 / len(sample_task)
            
        if 'itm' in sample_task:
            text_feature = features[:, :text_input_ids.size()[1], :]
            text_feature = text_feature[:, 0, :]
            pred = self.newfc_itm(text_feature)
            itm_loss = nn.BCEWithLogitsLoss()(pred.view(-1), video_text_match_label.view(-1))
            loss += itm_loss / 100 / len(sample_task)
         
        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return loss_category, accuracy, pred_label_id, label
    
    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    
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
    def forward(self, inputs, gather_index=None, return_mlm=False):
        encoder_outputs = self.bert(inputs)
        frame_len = inputs['frame_feature'].size()[1]
        if return_mlm:
            return encoder_outputs, self.cls(encoder_outputs)[:, :-frame_len, :]
        else:
            return encoder_outputs, None        
        
class UniBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        # self.video_fc = torch.nn.Linear(1536, config.hidden_size)
        self.video_embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, inputs, gather_index=None):
        frame_feature, frame_mask, frame_token_type_ids = inputs['frame_feature'], inputs['frame_mask'], inputs['frame_token_type_ids']
        text_input_ids, text_mask, text_token_type_ids = inputs['text_input_ids'], inputs['text_mask'], inputs['text_token_type_ids']
        text_emb = self.embeddings(input_ids=text_input_ids, token_type_ids=text_token_type_ids)
        frame_emb = self.video_embeddings(inputs_embeds=frame_feature, token_type_ids=frame_token_type_ids)

        embedding_output = torch.cat([text_emb, frame_emb], 1)
        mask = torch.cat([text_mask, frame_mask], 1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -1000000.0
        
        encoder_outputs = self.encoder(embedding_output, attention_mask=mask)['last_hidden_state']
        return encoder_outputs

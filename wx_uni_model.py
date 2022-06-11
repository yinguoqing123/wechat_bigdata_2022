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
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler
from data_helper import get_prior_lv1_lv2
from util import ArcFace, FocalLoss

class WXUniModel(nn.Module):
    def __init__(self, args, task=[], init_from_pretrain=True, use_arcface_loss=False):
        super().__init__()
        uni_bert_cfg = BertConfig.from_pretrained(args.bert_dir)
        #uni_bert_cfg.num_hidden_layers = 1
        
        self.frame_trans_layer = torch.nn.ModuleList([torch.nn.Linear(1024, uni_bert_cfg.hidden_size), 
                                  torch.nn.LayerNorm(uni_bert_cfg.hidden_size, eps=uni_bert_cfg.layer_norm_eps), torch.nn.Dropout(0.1)])
        self.dp = torch.nn.Dropout(0.1)
        self.frame_classify_aux1 = torch.nn.Linear(1024, 23)
        self.frame_classify_aux2 = torch.nn.Linear(1024, 200)
        self.use_arcface_loss = use_arcface_loss
        
        self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size, uni_bert_cfg, dropout=args.dropout)
        
        if not self.use_arcface_loss:
            self.classify_lv1 = torch.nn.Linear(768, 23)
            self.classify_lv2 = torch.nn.Linear(768, 200)
        else:
            self.classify_lv1 = ArcFace(768, 23, m=0.1)
            self.classify_lv2 = ArcFace(768, 200, m=0.05)
            
        self.prior_lv1, self.prior_lv2 = get_prior_lv1_lv2()
        self.prior_lv1, self.prior_lv2 = torch.tensor(self.prior_lv1, device='cuda', dtype=torch.float), torch.tensor(self.prior_lv2, device='cuda', dtype=torch.float)
        # self.classify_lv1.bias.data = self.prior_lv1
        # self.classify_lv2.bias.data = self.prior_lv2
    
        self.task = set(task)
        
        if 'mlm' in task:
            self.lm = MaskLM(tokenizer_path=args.bert_dir)
        
        if 'mvm' in task:
            self.vm = MaskVideo()
            self.roberta_mvm_lm_header = VisualOnlyMLMHead(uni_bert_cfg) 
            
        if 'itm' in task:
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(uni_bert_cfg.hidden_size, 1) 

        if init_from_pretrain:
            self.roberta = UniBertForMaskedLM.from_pretrained(args.bert_dir, config=uni_bert_cfg)
        else:
            self.roberta = UniBertForMaskedLM(uni_bert_cfg)

    def forward(self, inputs, target=None, task=None, inference=False, pretrain=False):
        loss, pred = 0, None
        frame_feature, frame_mask, frame_token_type_ids = inputs['frame_input'], inputs['frame_mask'], inputs['frame_token_type_ids']
        text_input_ids, text_mask, text_token_type_ids = inputs['text_input'], inputs['text_mask'], inputs['text_token_type_ids']
        
        frame_feature = self.nextvlad(frame_feature)
        frame_feature_for_classify = frame_feature
        for layer in self.frame_trans_layer:
            frame_feature = layer(frame_feature)
            
        frame_mask = torch.ones(frame_feature.shape[0], frame_feature.shape[1], dtype=torch.long, device=frame_feature.device)
        
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
        mask = torch.cat([text_mask, frame_mask], dim=-1).float()
        inputs =  dict(
            frame_input=frame_feature,
            frame_mask=frame_mask,
            frame_token_type_ids=frame_token_type_ids,
            text_input=text_input_ids,
            text_mask=text_mask,
            text_token_type_ids=text_token_type_ids,
            label=inputs.get('label', None),
            label_lv1=inputs.get(['label_lv1'], None)
        )
        features, lm_prediction_scores = self.roberta(inputs, return_mlm=return_mlm)
        features_mean_pooling = torch.sum(features * mask.unsqueeze(dim=-1), dim=1) / torch.sum(mask, dim=-1, keepdim=True)
        # features_max_pooling, _ = torch.max(features * mask.unsqueeze(dim=-1), dim=1)
        # features_mean_pooling = features[:, 0, :]
        # features_mean_pooling = torch.cat([features_mean_pooling, torch.dropout(frame_feature.squeeze(), p=0.1)], dim=-1)
        
        if not pretrain:
            if self.use_arcface_loss:
                prediction_lv1 = self.classify_lv1(features_mean_pooling, inputs['label_lv1'])
                prediction_lv2 = self.classify_lv2(features_mean_pooling, inputs['label'])
            else:
                prediction_lv1 = self.classify_lv1(features_mean_pooling)
                prediction_lv2 = self.classify_lv2(features_mean_pooling)
                prediction_lv1_only_frame = self.frame_classify_aux1(self.dp(frame_feature_for_classify.squeeze()))
                prediction_lv2_only_frame = self.frame_classify_aux2(frame_feature_for_classify.squeeze())
            # if not inference:
            prediction_lv1 = prediction_lv1 + self.prior_lv1.unsqueeze(dim=0) 
            prediction_lv2 = prediction_lv2 + self.prior_lv2.unsqueeze(dim=0)
        
        # compute loss
        
        if 'mlm' in sample_task:
            mlm_pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
            masked_lm_loss = nn.CrossEntropyLoss()(mlm_pred, lm_label.view(-1))
            loss += torch.log(masked_lm_loss + 1e-12)
            
        if 'mvm' in sample_task:
            vm_output = self.roberta_mvm_lm_header(features[:, text_input_ids.size()[1]:, :])
            masked_vm_loss = self.calculate_mfm_loss(vm_output, vm_input, 
                                                     frame_mask, video_label, normalize=False)
            loss += masked_vm_loss / 3 / len(sample_task)
            
        if 'itm' in sample_task:
            text_feature = features[:, 0, :]
            item_pred = self.newfc_itm(text_feature)
            itm_loss = nn.BCEWithLogitsLoss()(item_pred.view(-1), video_text_match_label.view(-1))
            loss += torch.log(itm_loss + 1e-12)
         
        if inference:
            return torch.argmax(prediction_lv2, dim=1)
        elif pretrain:
            return self.cal_pretrain_metric(loss, mlm_pred, lm_label, item_pred, video_text_match_label)
        else:
            return self.cal_loss(prediction_lv1, prediction_lv2, prediction_lv1_only_frame, prediction_lv2_only_frame, inputs['label_lv1'], inputs['label'], focal_loss=False)  # loss_category, accuracy, pred_label_id, label 
    
    @staticmethod
    def cal_loss(prediction_lv1, prediction_lv2, prediction_lv1_only_frame, prediction_lv2_only_frame, label_lv1, label_lv2, focal_loss=False):
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
            loss_lv1_only_frame = F.cross_entropy(prediction_lv1_only_frame, label_lv1, weight_lv1)
            loss_lv2_only_frame = F.cross_entropy(prediction_lv2_only_frame, label_lv2, weight_lv2)
        loss = loss_lv1 + loss_lv2 + loss_lv1_only_frame + loss_lv2_only_frame
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction_lv2, dim=1)
            pred_label_id_only_frame = torch.argmax(prediction_lv2_only_frame, dim=1)
            accuracy = (label_lv2 == pred_label_id).float().sum() / label_lv2.shape[0]
        return loss, accuracy, pred_label_id, label_lv2
    
    @staticmethod
    def cal_pretrain_metric(loss, mlm_pred, lm_label, itm_pred, itm_label):
        mlm_pred, itm_pred = mlm_pred.argmax(dim=-1), (itm_pred>0.5).int()
        mlm_pred, lm_label, itm_pred, item_label = mlm_pred.view(-1), lm_label.view(-1), itm_pred.view(-1), item_label.view(-1)
        accuracy_mlm = (mlm_pred == lm_label).float().sum() / (lm_label != -100).sum()
        accuracy_itm = (itm_pred == itm_label).float().sum() / itm_label.shape[0]
        return loss, accuracy_mlm, accuracy_itm
    
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
        frame_len = inputs['frame_input'].size()[1]
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
        frame_feature, frame_mask, frame_token_type_ids = inputs['frame_input'], inputs['frame_mask'], inputs['frame_token_type_ids']
        text_input_ids, text_mask, text_token_type_ids = inputs['text_input'], inputs['text_mask'], inputs['text_token_type_ids']
        # frame_feature = torch.tanh(self.video_fc(frame_feature))
        # frame_feature = self.video_fc1(frame_feature)
        text_emb = self.embeddings(input_ids=text_input_ids, token_type_ids=text_token_type_ids)
        # frame_emb = self.video_embeddings(inputs_embeds=frame_feature, token_type_ids=frame_token_type_ids)
        # frame_emb = self.embeddings(inputs_embeds=frame_feature, token_type_ids=frame_token_type_ids)
        frame_emb = frame_feature
        embedding_output = torch.cat([text_emb, frame_emb], 1)
        mask = torch.cat([text_mask, frame_mask], 1)
        mask = mask[:, None, None, :]
        mask = ((1.0 - mask) * -1000000.0).float()
        
        encoder_outputs = self.encoder(embedding_output, attention_mask=mask)['last_hidden_state']
        return encoder_outputs

class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, config, output_size=1024, expansion=2, groups=8, dropout=0.1):
        super().__init__()
        self.feature_size = feature_size
        self.expansion_size = expansion
        self.output_size = output_size
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout
        
        self.new_feature_size = self.expansion_size * self.feature_size // self.groups
        
        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = nn.Linear(self.cluster_size * self.new_feature_size, self.output_size)
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, inputs, mask=None):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, 1, self.cluster_size * self.new_feature_size])
        vlad = self.fc(vlad)
        return vlad

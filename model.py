import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import numpy as np, itertools, random, copy, math
from model_gcn import GCN, UNIMODALGCN

def print_grad(grad):
    print('the grad is', grad[2][0:5])
    return grad

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.5, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        # print(seq_length)
        # print(labels_length)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits,-1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()





def simple_batch_graphify(features, lengths):
    node_features = []
    
    batch_size = features.size(1)
    
    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])
    
    # for x in node_features:
        
    node_features = torch.cat(node_features, dim=0)  

    node_features = node_features.to("cuda:0")
    
    return node_features


        
class MultiHeadCrossModalAttention(nn.Module):
    def __init__(self, img_dim, txt_dim, hidden_dim, num_heads):
        super(MultiHeadCrossModalAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 多头投影层，将图像和文本分别映射到多头的 Q、K、V
        self.img_query_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_key_proj = nn.Linear(txt_dim, hidden_dim)
        self.txt_value_proj = nn.Linear(txt_dim, hidden_dim)

        # 最后的输出线性变换
        self.output_proj = nn.Linear(hidden_dim, img_dim)

    def forward(self, img_features, txt_features):
        """
        :param img_features: [batch_size, num_regions, img_dim]
        :param txt_features: [batch_size, num_words, txt_dim]
        :return: 融合后的特征
        """
        B, R, _ = img_features.shape  # B: batch_size, R: num_regions
        _, W, _ = txt_features.shape  # W: num_words

        # 线性投影得到 Q、K、V，并 reshape 为多头格式
        Q = self.img_query_proj(img_features).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)  
        K = self.txt_key_proj(txt_features).view(B, W, self.num_heads, self.head_dim).transpose(1, 2)  
        V = self.txt_value_proj(txt_features).view(B, W, self.num_heads, self.head_dim).transpose(1, 2)  

        # 计算注意力权重: Q·K^T / sqrt(d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  
        attention_weights = F.softmax(attention_scores, dim=-1)  

        # 加权求和得到上下文表示
        attended_features = torch.matmul(attention_weights, V)  

        # 合并多头的结果
        attended_features = attended_features.transpose(1, 2).contiguous().view(B, R, -1)  

        # 输出线性变换
        output = img_features + self.output_proj(attended_features)  
        return output


class Model(nn.Module):

    def __init__(self, 
                 base_model, 
                 D_m, D_g, D_e, 
                 graph_hidden_size, 
                 n_speakers,
                 n_classes=7, 
                 dropout=0.5,  
              
                 graph_type='relation',
                 alpha=0.2, 
                 use_residue=True,
                 D_m_v=512,
                 D_m_a=100,
                 modals='avl',
                 att_type='concat_DHT',
                 av_using_lstm=False,
                 dataset='IEMOCAP',
                 use_speaker=True,
                 num_L = 3,
                 num_K = 4,
                 modality_self_attention=False, 
                 original_gcn=False,
                 graph_masking=True):
        
        super(Model, self).__init__()
        self.original_gcn = original_gcn
        self.graph_masking = graph_masking
        self.modality_self_attention = modality_self_attention
        self.base_model = base_model
       
       
        self.graph_type=graph_type
        self.alpha = alpha
       
        self.dropout = dropout
        
        self.use_residue = use_residue
        
        self.return_feature = True
        self.modals = [x for x in modals]  # a, v, l
        self.use_speaker = use_speaker
        self.att_type = att_type
        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)

    
        self.multi_modal = True
        self.av_using_lstm = av_using_lstm
       
        self.dataset = dataset

        if self.base_model == 'LSTM':
            if not self.multi_modal:
                if len(self.modals) == 3:
                    hidden_ = 250
                elif ''.join(self.modals) == 'al':
                    hidden_ = 150
                elif ''.join(self.modals) == 'vl':
                    hidden_ = 150
                else:
                    hidden_ = 100
                self.linear_ = nn.Linear(D_m, hidden_)
                self.lstm = nn.LSTM(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            else:
                if 'a' in self.modals:
                    hidden_a = D_g
                    self.linear_a = nn.Linear(D_m_a, hidden_a)
                    if self.av_using_lstm:
                        self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'v' in self.modals:
                    hidden_v = D_g
                    self.linear_v = nn.Linear(D_m_v, hidden_v)
                    if self.av_using_lstm:
                        self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'l' in self.modals:
                    hidden_l = D_g
                    self.linear_l = nn.Linear(D_m, hidden_l)
                    self.lstm_l = nn.LSTM(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)

       
        if self.modality_self_attention:
            
            self.a_align = MultiHeadCrossModalAttention(D_g, D_g, D_g, 2) 
            self.v_align = MultiHeadCrossModalAttention(D_g, D_g, D_g, 2) 
            self.l_align = MultiHeadCrossModalAttention(D_g, D_g, D_g, 2) 
        
        else:
            self.align = MultiHeadCrossModalAttention(D_g, D_g, D_g, 2) 

        self.graph_model = GCN(n_dim=D_g, nhidden=graph_hidden_size, 
                                        dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True, return_feature=self.return_feature, use_residue=self.use_residue, n_speakers=n_speakers, modals=self.modals, use_speaker=self.use_speaker,  num_L=num_L, num_K=num_K, original_gcn=self.original_gcn, graph_masking=self.graph_masking)
        print("construct "+self.graph_type)

        if self.multi_modal:
            self.dropout_ = nn.Dropout(self.dropout)
            self.hidfc = nn.Linear(graph_hidden_size, n_classes)
          
            if self.att_type == 'concat_DHT':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size*2)*len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear((graph_hidden_size*2)*len(self.modals), n_classes)
          


    # def _reverse_seq(self, X, mask):
    #     X_ = X.transpose(0,1)
    #     mask_sum = torch.sum(mask, 1).int()

    #     xfs = []
    #     for x, c in zip(X_, mask_sum):
    #         xf = torch.flip(x[:c], [0])
    #         xfs.append(xf)

    #     return pad_sequence(xfs)


    def forward(self, U, qmask, umask, seq_lengths, U_a=None, U_v=None, epoch=None):

        #=============roberta features
        [r1,r2,r3,r4]=U
        seq_len, _, feature_dim = r1.size()

        r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

        U = (r1 + r2 + r3 + r4)/4
       
        if self.base_model == 'LSTM':
            if not self.multi_modal:
                if self.modals == "l":
                    U = self.linear_l(U)
                    emotions, _ = self.lstm(U)
                elif self.modals == "v":
                    U_v = self.linear_v(U_v)
                    emotions = U_v
                elif self.modals == "a":
                    U_a = self.linear_a(U_a)
                    emotions = U_a
            else:
                if self.av_using_lstm:
                    if 'a' in self.modals:
                        
                        U_a = self.linear_a(U_a)
                        
                        emotions_a, _ = self.lstm_a(U_a)
                        
                    if 'v' in self.modals:
                        U_v = self.linear_v(U_v)
                        emotions_v, _ = self.lstm_v(U_v)
                else:
                    
                    emotions_a = self.linear_a(U_a)
                    
                    emotions_v = self.linear_v(U_v)

                if 'l' in self.modals:
                    
                    U = self.linear_l(U)
                    
                    emotions_l, _ = self.lstm_l(U)
                    
                    
                if self.modality_self_attention:
                    emotions_a = self.a_align(emotions_a, emotions_a)
                    emotions_v = self.v_align(emotions_v, emotions_v)
                    emotions_l = self.l_align(emotions_l, emotions_l)
                
                else:
                    emotions_a = self.align(emotions_a, emotions_l) 
                    emotions_v = self.align(emotions_v, emotions_l) 
                    
        if not self.multi_modal:
            
            features = simple_batch_graphify(emotions, seq_lengths)
        else:
            if 'a' in self.modals:
                features_a = simple_batch_graphify(emotions_a, seq_lengths)
                
            else:
                features_a = []
            if 'v' in self.modals:
                features_v = simple_batch_graphify(emotions_v, seq_lengths)
            else:
                features_v = []
            if 'l' in self.modals:
                
                
                features_l = simple_batch_graphify(emotions_l, seq_lengths)
                
                
            else:
                features_l = []
        if self.graph_type=='hyper':
            emotions_feat = self.graph_model(features_a, features_v, features_l, seq_lengths, qmask, epoch)
            
            emotions_feat = self.dropout_(emotions_feat)
            
            emotions_feat = nn.ReLU()(emotions_feat)
            
            
            # print(emotions_feat.shape)
            log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
            
        else:
            print("There are no such kind of graph")   
            
        
        return log_prob



import torch
import torch.nn as nn
import torch.nn.functional as F
from model_gcn import GCN

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
                 D_m,
                 D_g, 
                 graph_hidden_size, 
                 n_speakers,
                 n_classes=7, 
                 dropout=0.5,  
                 D_m_v=512,
                 D_m_a=100,
                 num_graph_layers = 4,
                 graph_masking=True, 
                 spk_embs = None,
                 using_lstms = None,
                 aligns = None, 
                 MRL = None, 
                 MRL_efficient = None,
                 num_MRL_partition = None,
                 num_heads = None,
                 mask_prob =None,
                 MKD = None, 

                 num_graph_layers_a=4,
                 num_graph_layers_v=4,
                 num_graph_layers_t=4,
                 graph_masking_a=True,
                 graph_masking_v=True,
                 graph_masking_t=True,
                 spk_embs_uni_modal_a=True,
                 spk_embs_uni_modal_v=True,
                 spk_embs_uni_modal_t=True,
                 lstm_unimodal_a=True,
                 lstm_unimodal_v=True,
                 lstm_unimodal_t=True,
                 aligns_uni_modal_a=True,
                 aligns_uni_modal_v=True,
                 aligns_uni_modal_t=True,
                 num_heads_a=2,
                 num_heads_v=2,
                 num_heads_t=2,
                 mask_prob_a=0.5,
                 mask_prob_v=0.5,
                 mask_prob_t=0.5,
                 MKD_a_layer = 0,
                 MKD_v_layer = 0,
                 MKD_t_layer = 0
                 ):
        
        super(Model, self).__init__()
        self.MKD_a_layer = MKD_a_layer
        self.MKD_v_layer = MKD_v_layer
        self.MKD_t_layer = MKD_t_layer
        self.num_graph_layers_a=num_graph_layers_a
        self.graph_masking_a=graph_masking_a
        self.spk_embs_uni_modal_a=spk_embs_uni_modal_a
        self.lstm_unimodal_a=lstm_unimodal_a
        self.aligns_uni_modal_a=aligns_uni_modal_a
        self.num_heads_a=num_heads_a
        self.mask_prob_a=mask_prob_a
        
        
        self.num_graph_layers_v=num_graph_layers_v
        self.graph_masking_v=graph_masking_v
        self.spk_embs_uni_modal_v=spk_embs_uni_modal_v
        self.lstm_unimodal_v=lstm_unimodal_v
        self.aligns_uni_modal_v=aligns_uni_modal_v
        self.num_heads_v=num_heads_v
        self.mask_prob_v=mask_prob_v
        
        
        self.num_graph_layers_t=num_graph_layers_t
        self.graph_masking_t=graph_masking_t
        self.spk_embs_uni_modal_t=spk_embs_uni_modal_t
        self.lstm_unimodal_t=lstm_unimodal_t
        self.aligns_uni_modal_t=aligns_uni_modal_t
        self.num_heads_t=num_heads_t
        self.mask_prob_t=mask_prob_t
        
        
        
        self.MKD = MKD
        self.mask_prob = mask_prob
        self.num_heads = num_heads
        self.MRL = MRL
        self.MRL_efficient = MRL_efficient
        self.num_MRL_partition = num_MRL_partition
        
        
        self.spk_embs = spk_embs
        self.using_lstms = using_lstms
        self.aligns = aligns
        
        self.graph_masking = graph_masking

       
       
        
       
        self.dropout = dropout
        
        
        self.return_feature = True
        
        
        
        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)

    
       
   
        
        hidden_a = D_g
        hidden_v = D_g
        hidden_t = D_g

        self.linear_a = nn.Linear(D_m_a, hidden_a)
        if "a" in self.using_lstms:
            self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=self.dropout)
    
        
        self.linear_v = nn.Linear(D_m_v, hidden_v)
        
        if "v" in self.using_lstms:
            self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=self.dropout)
    
        
        self.linear_t = nn.Linear(D_m, hidden_t)
        
        if "t" in self.using_lstms:
            self.lstm_t = nn.LSTM(input_size=hidden_t, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=self.dropout)

 
        self.align = MultiHeadCrossModalAttention(D_g, D_g, D_g, num_heads=self.num_heads) 

        self.graph_model = GCN(n_dim=D_g, 
                               nhidden=graph_hidden_size,
                               dropout=self.dropout,
                               return_feature=self.return_feature,
                               n_speakers=n_speakers, 
                               num_graph_layers=num_graph_layers,
                               graph_masking=self.graph_masking,
                               spk_embs = self.spk_embs,
                               mask_prob= self.mask_prob,
                               MKD = self.MKD, 
                               MKD_a_layer=self.MKD_a_layer ,
                               MKD_v_layer=self.MKD_v_layer ,
                               MKD_t_layer=self.MKD_t_layer )

        
        self.dropout_ = nn.Dropout(self.dropout)
        self.smax_fc = nn.Linear((graph_hidden_size*2)*3, n_classes)
        
        
        if self.MKD == True:
            print("MKD 시작")
            self.unimodel_a = UniModel("a",
                                  D_m=D_m,
                                  D_g=D_g, 
                                  graph_hidden_size=graph_hidden_size,
                                  n_speakers=n_speakers,
                                  n_classes=n_classes, 
                                  dropout=dropout,
                                  D_m_v=D_m_v,
                                  D_m_a=D_m_a,
                                  num_graph_layers=self.num_graph_layers_a,
                                  graph_masking=self.graph_masking_a, 
                                  spk_embs_uni_modal = self.spk_embs_uni_modal_a,
                                  lstm_unimodal = self.lstm_unimodal_a,
                                  aligns_uni_modal = self.aligns_uni_modal_a, 
                                  num_heads = self.num_heads_a,
                                  mask_prob =self.mask_prob_a)
            self.unimodel_v = UniModel("v",
                                  D_m=D_m,
                                  D_g=D_g, 
                                  graph_hidden_size=graph_hidden_size,
                                  n_speakers=n_speakers,
                                  n_classes=n_classes, 
                                  dropout=dropout,
                                  D_m_v=D_m_v,
                                  D_m_a=D_m_a,
                                  num_graph_layers=self.num_graph_layers_v,
                                  graph_masking=self.graph_masking_v, 
                                  spk_embs_uni_modal = self.spk_embs_uni_modal_v,
                                  lstm_unimodal = self.lstm_unimodal_v,
                                  aligns_uni_modal = self.aligns_uni_modal_v, 
                                  num_heads = self.num_heads_v,
                                  mask_prob =self.mask_prob_v)
            self.unimodel_t = UniModel("t",
                                  D_m=D_m,
                                  D_g=D_g, 
                                  graph_hidden_size=graph_hidden_size,
                                  n_speakers=n_speakers,
                                  n_classes=n_classes, 
                                  dropout=dropout,
                                  D_m_v=D_m_v,
                                  D_m_a=D_m_a,
                                  num_graph_layers=self.num_graph_layers_t,
                                  graph_masking=self.graph_masking_t, 
                                  spk_embs_uni_modal = self.spk_embs_uni_modal_t,
                                  lstm_unimodal = self.lstm_unimodal_t,
                                  aligns_uni_modal = self.aligns_uni_modal_t, 
                                  num_heads = self.num_heads_t,
                                  mask_prob =self.mask_prob_t)
            self.smax_fc_t = nn.Linear(graph_hidden_size*2, n_classes)
            self.smax_fc_v = nn.Linear(graph_hidden_size*2, n_classes)
            self.smax_fc_a = nn.Linear(graph_hidden_size*2, n_classes)
        
        
        if self.MRL == True:
            unit_hiddensizes = []
            unit_temp_hiddensize = (graph_hidden_size*2)
            for _ in range(self.num_MRL_partition):
                unit_temp_hiddensize = unit_temp_hiddensize // 2
                unit_hiddensizes.append(unit_temp_hiddensize)
                
            self.unit_hiddensizes = unit_hiddensizes
            
            
            if self.MRL_efficient:
                pass
            
            else:
                self.last_layers = nn.ModuleList()
                for unit_hiddensize in self.unit_hiddensizes:
                    self.last_layers.append(nn.Linear(3 * unit_hiddensize, n_classes))
                    
            

    def forward(self, U, qmask, seq_lengths, U_a=None, U_v=None, epoch=None):

        if self.MKD:
            emotions_feat_uni_a = self.unimodel_a(U, qmask, seq_lengths, U_a, U_v, epoch)
            emotions_feat_uni_v = self.unimodel_v(U, qmask, seq_lengths, U_a, U_v, epoch)
            emotions_feat_uni_t = self.unimodel_t(U, qmask, seq_lengths, U_a, U_v, epoch)
        
        
        
        #=============roberta features
        [r1,r2,r3,r4]=U
        seq_len, _, feature_dim = r1.size()

        r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

        U_t = (r1 + r2 + r3 + r4)/4
        
        U_a = self.linear_a(U_a)        
        U_v = self.linear_v(U_v)
        U_t = self.linear_t(U_t)
        
        # print(U_a.size())
        if "a" in self.using_lstms:
            emotions_a, _ = self.lstm_a(U_a)
        else:
            emotions_a = U_a
        if "v" in self.using_lstms:
            emotions_v, _ = self.lstm_v(U_v)
        else:
            emotions_v = U_v
        if "t" in self.using_lstms:
            emotions_t, _ = self.lstm_t(U_t)
        else:
            emotions_t = U_t
        
        if self.aligns=="to_t":
            
            emotions_a = self.align(emotions_a, emotions_t) 
            emotions_v = self.align(emotions_v, emotions_t) 
    
        elif self.aligns=="to_v":
            emotions_a = self.align(emotions_a, emotions_v)
            emotions_t = self.align(emotions_t, emotions_v)
            
        elif self.aligns == "to_a":
            emotions_v = self.align(emotions_v, emotions_a)
            emotions_t = self.align(emotions_t, emotions_a)
            
        elif self.aligns == "NO":
            pass
            
        # print("graph 들어가기전:",emotions_a.size())
            
        features_a = simple_batch_graphify(emotions_a, seq_lengths)
        features_v = simple_batch_graphify(emotions_v, seq_lengths)
        features_t = simple_batch_graphify(emotions_t, seq_lengths)
        
        if self.MKD:
            emotions_feat, emotions_feat_MM_a, emotions_feat_MM_v, emotions_feat_MM_t = self.graph_model(features_a, features_v, features_t, seq_lengths, qmask)   
           
        else:
            emotions_feat = self.graph_model(features_a, features_v, features_t, seq_lengths, qmask)   
        
        # print(emotions_feat.size())     
        emotions_feat = self.dropout_(emotions_feat)        
        emotions_feat = nn.ReLU()(emotions_feat)
        
        
        
        
        
        if self.MRL == True:
            output_log_probs = []
            uni_modality_length = emotions_feat.shape[-1]//3
            if self.training:
                for k, hiddensize in enumerate(self.unit_hiddensizes):
                    x_selected = torch.cat([emotions_feat[:,i*uni_modality_length:i*uni_modality_length+hiddensize] for i in range(3)], dim=-1)
                    if self.MRL_efficient:
                        weight_selected = torch.cat([self.smax_fc.weight[:,i*uni_modality_length:i*uni_modality_length+hiddensize] for i in range(3)], dim=1)
                        output_log_probs.append(F.log_softmax(x_selected@weight_selected.T+self.smax_fc.bias,1))
                    else:                    
                        output_log_probs.append(F.log_softmax(self.last_layers[k](x_selected), 1))
                
                log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
                return output_log_probs, log_prob
            else:
                log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
                return log_prob
        
        else: 
            log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
            
            
            if self.MKD:
                if self.training:
                    # print(log_prob.size())  
                    log_prob_a_teacher = emotions_feat_uni_a      
                    log_prob_v_teacher = emotions_feat_uni_v      
                    log_prob_t_teacher = emotions_feat_uni_t      
                    
                    # print("teacher embedding size: ", log_prob_a_teacher.size())
                    length_emotions_feat = emotions_feat.shape[-1]
                    uni_feat_length = length_emotions_feat//3
                    emotions_feat_t = emotions_feat_MM_t[:, 0:uni_feat_length]
                    # print(emotions_feat.shape)
                    # print(emotions_feat_l.shape)
                    emotions_feat_a = emotions_feat_MM_a[:, uni_feat_length:2*uni_feat_length]
                    emotions_feat_v = emotions_feat_MM_v[:, 2*uni_feat_length:]
                    
                    score_t = self.smax_fc_t(emotions_feat_t) 
                    score_a = self.smax_fc_a(emotions_feat_a)
                    score_v = self.smax_fc_v(emotions_feat_v)
                        
                    log_prob_t = F.log_softmax(score_t,1)
                    log_prob_a = F.log_softmax(score_a, 1)
                    log_prob_v = F.log_softmax(score_v, 1)
                    
                    return log_prob, log_prob_a,log_prob_v,log_prob_t, log_prob_a_teacher, log_prob_v_teacher, log_prob_t_teacher
                else:
                    return log_prob

            else:
                return log_prob
        
        
        
        
class UniModel(nn.Module):

    def __init__(self, 
                 uni_modal,
                 D_m,
                 D_g, 
                 graph_hidden_size,
                 n_speakers,
                 n_classes=7, 
                 dropout=0.5,  
                 D_m_v=512,
                 D_m_a=100,
                 num_graph_layers = 4,
                 graph_masking=True, 
                 spk_embs_uni_modal = None,
                 lstm_unimodal = None,
                 aligns_uni_modal = None, 

                 num_heads = None,
                 mask_prob =None):
    
            
        super(UniModel, self).__init__()
        
        self.uni_modal = uni_modal
        self.mask_prob = mask_prob
        self.num_heads = num_heads
        
        
        self.lstm_unimodal = lstm_unimodal
        self.aligns_uni_modal = aligns_uni_modal
        
        self.graph_masking= graph_masking

       
        self.spk_embs_uni_modal = spk_embs_uni_modal
        
       
        self.dropout = dropout
        
        
        self.return_feature = True
        
        hidden_D = D_g
        if self.uni_modal== "t":
            self.normBNa = nn.BatchNorm1d(1024, affine=True)
            self.normBNb = nn.BatchNorm1d(1024, affine=True)
            self.normBNc = nn.BatchNorm1d(1024, affine=True)
            self.normBNd = nn.BatchNorm1d(1024, affine=True)

        if self.uni_modal == "a":
            self.linear = nn.Linear(D_m_a, hidden_D)
            
        elif self.uni_modal == "v":
            self.linear = nn.Linear(D_m_v, hidden_D)
        elif self.uni_modal == "t":
            self.linear = nn.Linear(D_m, hidden_D)
            
        if self.lstm_unimodal:
            self.lstm = nn.LSTM(input_size=hidden_D, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=self.dropout)

        if self.aligns_uni_modal:
            self.align = MultiHeadCrossModalAttention(D_g, D_g, D_g, num_heads=self.num_heads) 

        self.graph_model = GCN(n_dim=D_g, 
                               nhidden=graph_hidden_size,
                               dropout=self.dropout,
                               return_feature=self.return_feature,
                               n_speakers=n_speakers, 
                               num_graph_layers=num_graph_layers,
                               graph_masking=self.graph_masking,
                               spk_embs = None,
                               mask_prob= self.mask_prob, 
                               uni_modal = self.uni_modal,
                               spk_embs_uni_modal = self.spk_embs_uni_modal)

        
        self.dropout_ = nn.Dropout(self.dropout)
        self.smax_fc = nn.Linear((graph_hidden_size*2), n_classes)
        
       
    def forward(self, U, qmask, seq_lengths, U_a=None, U_v=None, epoch=None):

        #=============roberta features
        
        if self.uni_modal == "t":
            [r1,r2,r3,r4]=U
            seq_len, _, feature_dim = r1.size()

            r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

            U = (r1 + r2 + r3 + r4)/4
            U = self.linear(U)
            
        if self.uni_modal == "a":
            U = self.linear(U_a)        
            
        elif self.uni_modal == "v":
            U = self.linear(U_v)

        
        
        if self.lstm_unimodal:
        
            emotions, _ = self.lstm(U)
        else:
            emotions = U
        
        
        if self.aligns_uni_modal:
            
            emotions = self.align(emotions, emotions) 
            
        # print("unimodal graph 들어가기전", emotions.size())
        features = simple_batch_graphify(emotions, seq_lengths)
            
        emotions_feat = self.graph_model(features, features, features, seq_lengths, qmask)        
        emotions_feat = self.dropout_(emotions_feat)        
        emotions_feat = nn.ReLU()(emotions_feat)
        
        log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
                
        return log_prob



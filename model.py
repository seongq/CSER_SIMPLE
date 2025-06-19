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
                 graph_masking=True):
        
        super(Model, self).__init__()
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
        self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=self.dropout)
    
        
        self.linear_v = nn.Linear(D_m_v, hidden_v)
        self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=self.dropout)
    
        
        self.linear_t = nn.Linear(D_m, hidden_t)
        self.lstm_t = nn.LSTM(input_size=hidden_t, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=self.dropout)

 
        self.align = MultiHeadCrossModalAttention(D_g, D_g, D_g, 2) 

        self.graph_model = GCN(n_dim=D_g, 
                               nhidden=graph_hidden_size,
                               dropout=self.dropout,
                               return_feature=self.return_feature,
                               n_speakers=n_speakers, 
                               num_graph_layers=num_graph_layers,
                               graph_masking=self.graph_masking)

        
        self.dropout_ = nn.Dropout(self.dropout)
        self.smax_fc = nn.Linear((graph_hidden_size*2)*3, n_classes)
        


    def forward(self, U, qmask, seq_lengths, U_a=None, U_v=None, epoch=None):

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
        
        emotions_a, _ = self.lstm_a(U_a)
        emotions_v, _ = self.lstm_v(U_v)
        emotions_t, _ = self.lstm_t(U_t)
            
        emotions_a = self.align(emotions_a, emotions_t) 
        emotions_v = self.align(emotions_v, emotions_t) 
    
        features_a = simple_batch_graphify(emotions_a, seq_lengths)
        features_v = simple_batch_graphify(emotions_v, seq_lengths)
        features_t = simple_batch_graphify(emotions_t, seq_lengths)
            
        
        emotions_feat = self.graph_model(features_a, features_v, features_t, seq_lengths, qmask)        
        emotions_feat = self.dropout_(emotions_feat)        
        emotions_feat = nn.ReLU()(emotions_feat)
        log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
            
     
        
        return log_prob



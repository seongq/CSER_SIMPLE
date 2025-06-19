import torch 
import torch.nn as nn
from itertools import permutations
from graphgcn import GraphGCN


class GCN(nn.Module):
    def __init__(self, 
                 n_dim, 
                 nhidden,
                 dropout,
                 return_feature,
                 n_speakers=2,
                 modals=['a','v','l'],
                 use_speaker=True,
                 num_graph_layers=4,
                 original_gcn=False,
                 graph_masking=True):
        super(GCN, self).__init__()
        self.return_feature = return_feature  #True
        

        self.original_gcn = original_gcn
        self.graph_masking = graph_masking
        
        
        self.dropout = dropout
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        #------------------------------------    
        self.fc1 = nn.Linear(n_dim, nhidden)         
        self.num_graph_layers =  num_graph_layers
        
        for kk in range(self.num_graph_layers):
            setattr(self,'conv%d' %(kk+1), GraphGCN(nhidden, nhidden,  graph_masking=self.graph_masking))

    def forward(self, a, v, l, dia_len, qmask, epoch):
        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)
        
        
        l += spk_emb_vector

        a += spk_emb_vector

        v += spk_emb_vector

           
        
        
        gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len, self.modals)
        x1 = self.fc1(gnn_features)  
        out = x1
        gnn_out = x1
        for kk in range(self.num_graph_layers):
            gnn_out = gnn_out + getattr(self,'conv%d' %(kk+1))(gnn_out,gnn_edge_index)

        out2 = torch.cat([out,gnn_out], dim=1)
        
        
        out1 = self.reverse_features(dia_len, out2)
        
        return out1

    def reverse_features(self, dia_len, features):
        l=[]
        a=[]
        v=[]
        for i in dia_len:
            ll = features[0:1*i]
            aa = features[1*i:2*i]
            vv = features[2*i:3*i]
            features = features[3*i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)
        tmpl = torch.cat(l,dim=0)
        tmpa = torch.cat(a,dim=0)
        tmpv = torch.cat(v,dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features


    def create_gnn_index(self, a, v, l, dia_len, modals):
        num_modality = len(modals)
        node_count = 0
        index =[]
        tmp = []
        
        
        for i in dia_len:
            nodes = list(range(i*num_modality))
            
            nodes = [j + node_count for j in nodes] 
            
            nodes_l = nodes[0:i*num_modality//3]
            nodes_a = nodes[i*num_modality//3:i*num_modality*2//3]
            nodes_v = nodes[i*num_modality*2//3:]
            index = index + list(permutations(nodes_l,2)) + list(permutations(nodes_a,2)) + list(permutations(nodes_v,2))
            
            Gnodes=[]
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
                
            for ii, _ in enumerate(Gnodes):
                tmp = tmp +  list(permutations(_,2))
                
            if node_count == 0:
                ll = l[0:0+i]
                
                aa = a[0:0+i]
                vv = v[0:0+i]
                features = torch.cat([ll,aa,vv],dim=0)
                temp = 0+i
            else:
                
                ll = l[temp:temp+i]
                aa = a[temp:temp+i]
                vv = v[temp:temp+i]
                features_temp = torch.cat([ll,aa,vv],dim=0)
                features =  torch.cat([features,features_temp],dim=0)
                temp = temp+i
            node_count = node_count + i*num_modality
        edge_index = torch.cat([torch.LongTensor(index).T,torch.LongTensor(tmp).T],1).to("cuda:0")
        
        return edge_index, features
    
    
    
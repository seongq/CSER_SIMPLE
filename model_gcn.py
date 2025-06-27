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
                 num_graph_layers=4,
                 graph_masking=True,
                 spk_embs = None,
                 mask_prob = None,
                 uni_modal = None,
                 spk_embs_uni_modal = None,
                 MKD = None,
                 MKD_a_layer=None,
                MKD_v_layer=None,
                MKD_t_layer=None,
                 
                 ):
        super(GCN, self).__init__()
        
        self.MKD = MKD
        self.MKD_a_layer = MKD_a_layer
        self.MKD_v_layer = MKD_v_layer
        self.MKD_t_layer = MKD_t_layer
        
        self.uni_modal = uni_modal
        self.spk_embs_uni_modal = spk_embs_uni_modal
        
        
        self.mask_prob = mask_prob
        self.spk_embs = spk_embs
        self.return_feature = return_feature  #True
        

        self.graph_masking = graph_masking
        
        
        self.dropout = dropout
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        #------------------------------------    
        self.fc1 = nn.Linear(n_dim, nhidden)         
        self.num_graph_layers =  num_graph_layers
        
        for kk in range(self.num_graph_layers):
            setattr(self,'conv%d' %(kk+1), GraphGCN(nhidden, nhidden,  graph_masking=self.graph_masking, mask_prob = self.mask_prob))

    def forward(self, a, v, l, dia_len, qmask):
        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)
        
        if self.uni_modal:
            
            if self.spk_embs_uni_modal:
                if self.uni_modal== "a":
                    a += spk_emb_vector
                    
                elif self.uni_modal == "v":
                    v += spk_emb_vector
                elif self.uni_modal == "t":
                    l += spk_emb_vector    
                
                
                else:
                    raise ValueError(f"Invalid uni_modal value: {self.uni_modal}. Must be one of ['a', 'v', 't']")
        
        else:
            if "t" in self.spk_embs:
                l += spk_emb_vector
            elif "a" in self.spk_embs:
                a += spk_emb_vector
            elif "v" in self.spk_embs:
                v += spk_emb_vector

           
        
        
        gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len)
        # print(gnn_features.size())
        
        
        
        if (not self.uni_modal) and self.MKD:
            outputs = {}
            outputs['total']=None
            outputs['a_layer']=None
            outputs['v_layer']=None
            outputs['t_layer']=None
            #gnn_features 는 -1, torch.cat([gnn_features,gnn_features])
           
            if self.MKD_a_layer == -1:
                outputs['a_layer'] = self.reverse_features(dia_len, gnn_features)
            if self.MKD_v_layer == -1:
                outputs['v_layer'] = self.reverse_features(dia_len, gnn_features)
            if self.MKD_t_layer == -1:
                outputs['t_layer'] = self.reverse_features(dia_len, gnn_features)
          
            x1 = self.fc1(gnn_features)  #x1은 0 , torch.cat([x1, out])
            # print(gnn_features.size())
            
            out = x1
            # print(out.size())
            gnn_out = x1
            
            if self.MKD_a_layer == 0:
                outputs['a_layer'] = self.reverse_features(dia_len, torch.cat([out,gnn_out], dim=1))
            if self.MKD_v_layer == 0:
                outputs['v_layer'] = self.reverse_features(dia_len, torch.cat([out,gnn_out], dim=1))
            if self.MKD_t_layer == 0:
                outputs['t_layer'] = self.reverse_features(dia_len, torch.cat([out,gnn_out], dim=1))
            
            
            
            for kk in range(self.num_graph_layers):
                gnn_out = gnn_out + getattr(self,'conv%d' %(kk+1))(gnn_out,gnn_edge_index)
                if self.MKD_a_layer == kk+1:
                    outputs['a_layer'] = self.reverse_features(dia_len, torch.cat([out,gnn_out], dim=1))
                if self.MKD_v_layer == kk+1:
                    outputs['v_layer'] = self.reverse_features(dia_len, torch.cat([out,gnn_out], dim=1))
                if self.MKD_t_layer == kk+1:
                    outputs['t_layer'] = self.reverse_features(dia_len, torch.cat([out,gnn_out], dim=1))
       
            out2 = torch.cat([out,gnn_out], dim=1)
            
            out1 = self.reverse_features(dia_len, out2)
            outputs['total'] = out1
            return outputs['total'], outputs['a_layer'], outputs['v_layer'], outputs['t_layer']     
        else:
            #gnn_features 는 -1, torch.cat([gnn_features,gnn_features])
        
            
            x1 = self.fc1(gnn_features)  #x1은 0 , torch.cat([x1, out])
        
            out = x1
            gnn_out = x1
            for kk in range(self.num_graph_layers):
                gnn_out = gnn_out + getattr(self,'conv%d' %(kk+1))(gnn_out,gnn_edge_index)
               
            out2 = torch.cat([out,gnn_out], dim=1)
            
            
            out1 = self.reverse_features(dia_len, out2)
            
            return out1

    def reverse_features(self, dia_len, features):
        if self.uni_modal:
            tmplist = []
            for i in dia_len:
                tmpfeatures = features[0:1*i]
                features = features[1*i:]
                tmplist.append(tmpfeatures)
            features = torch.cat(tmplist,dim=0)
        
        else:
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


    def create_gnn_index(self, a, v, l, dia_len):
        
        if self.uni_modal:
            node_count = 0
            index =[]
            tmp = []
            if self.uni_modal == "t":
                uni_feature = l
            elif self.uni_modal == "a":
                uni_feature = a
            elif self.uni_modal == "v":
                uni_feature = v
            for i in dia_len:
                nodes = list(range(i))
                
                nodes = [j + node_count for j in nodes] 
                
                index = index + list(permutations(nodes,2)) 
                
                if node_count == 0:
                    features = uni_feature[0:0+i]
                    temp = 0+i
                else:
                    features_temp = uni_feature[temp:temp+i]                
                    features =  torch.cat([features,features_temp],dim=0)
                    temp = temp+i
                node_count = node_count + i
            edge_index = torch.LongTensor(index).T.to("cuda:0")
        
        
        else:
            num_modality = 3
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
                    
                for _, _ in enumerate(Gnodes):
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
    
    
    
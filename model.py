from turtle import forward
import torch
import torch.nn as nn
import esm
import copy
import math
from einops import rearrange, repeat

class BaseClsModel(nn.Module):
    """
    Cls model for protein binding affinity
    """
    def __init__(self, n_embedding=1280, n_hidden=50, n_classes=1):
        super(BaseClsModel, self).__init__()
        self.model_name = 'BaseClsModel'
        self.n_embedding = n_embedding
        self.n_classes = n_classes
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding*3, n_hidden),
            nn.ReLU(),
#            nn.Dropout(0.5),
            nn.Linear(n_hidden, n_classes)
        )

    def forward(self,data):
        out = self.classifier(data)
        out = out.squeeze(dim=-1)
        return out

class AntibactCLSModel(nn.Module):
    """
    A model for antibact classfication
    """
    def __init__(self,n_embedding=1280,n_hidden=768,n_classes=1):
        super(AntibactCLSModel,self).__init__()
        self.ProteinBert, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding, n_hidden),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(n_hidden, 80),
            nn.ReLU(),
            nn.Linear(80,n_classes)
        )

    def forward(self, data):
        results = self.ProteinBert(data, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        cls_embedding = token_representations[:,0,:]  # cls token
        out = self.classifier(cls_embedding)
        out = out.squeeze(dim=-1)
        return out

class AntibactRegModel(nn.Module):
    """
    A model for antibact regression
    """
    def __init__(self,n_embedding=1280):
        super(AntibactRegModel,self).__init__()
        self.ProteinBert, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.Predictor = nn.Sequential(
            nn.Linear(n_embedding, 500),
            nn.ReLU(),
            nn.Linear(500,100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Linear(50,1)
        )

    def forward(self, data):
        results = self.ProteinBert(data, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        cls_embedding = token_representations[:,0,:]  # cls token
        out = self.Predictor(cls_embedding)
        out = out.squeeze(dim=-1)
        return out
        
class AntibactRankingModel(nn.Module):
    """
    Model for antibact ranking
    """
    def __init__(self, n_embedding=1280, n_classes=1):
        super(AntibactRankingModel, self).__init__()
        self.model_name = 'RankingModel'
        self.n_embedding = n_embedding
        self.n_classes = n_classes
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding*3, n_embedding),
            nn.ReLU(),
            nn.Linear(n_embedding, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes)
        )

    def forward(self,data):
        out = self.classifier(data)
        out = out.squeeze(dim=-1)
        return out

class DenoisingAE(nn.Module):
    def __init__(self):
        super(DenoisingAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(676,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,676)
        )

    def forward(self, x, inference=False):
        x = self.encoder(x)

        if not inference:
            x = self.decoder(x)
        return x

class NormalMLP(nn.Module):

    def __init__(self, n_embedding=676, n_classes=1):
        super(NormalMLP, self).__init__()
        self.model_name = 'NormalMLP'
        self.n_embedding = n_embedding
        self.n_classes = n_classes
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding,256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self,data):
        out = self.classifier(data)
        out = out.squeeze(dim=-1)
        return out

class Ranking_earlyfusion(nn.Module):
    """
    A early fusion version of Multimodal ranking model 
    """
    def __init__(self,n_embedding=256+128):
        super(Ranking_earlyfusion,self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(1), requires_grad=True)
        self.emb_layers = nn.Sequential(
             nn.Linear(1280, 512),
             nn.ReLU(),
             nn.Linear(512, 256),
             nn.ReLU(),
        )
        self.structured_layers = nn.Sequential(
             nn.Linear(676, 256),
             nn.ReLU(),
             nn.Linear(256, 128),
             nn.ReLU(),
        )
        self.Predictor = nn.Sequential(
            nn.Linear(n_embedding, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self, emb, structure_feature):
        emb = self.emb_layers(emb)
        enc = self.structured_layers(structure_feature)
        enc = enc*self.weights
        feature = torch.cat((emb,enc),dim=1)
        out = self.Predictor(feature)
        out = out.squeeze(dim=-1)
        return out

class Ranking_GateFusion(nn.Module):
    """
    A early fusion version of Multimodal ranking model 
    """
    def __init__(self,feature_size=256, hidden_size=128):
        super(Ranking_GateFusion,self).__init__()
        self.emb_layers = nn.Sequential(
             nn.Linear(1280, 512),
             nn.ReLU(),
             nn.Linear(512, feature_size),
             nn.ReLU(),
        )
        self.structured_layers = nn.Sequential(
             nn.Linear(676, 512),
             nn.ReLU(),
             nn.Linear(512, feature_size),
             nn.ReLU(),
        )
        self.e2h = nn.Linear(feature_size,hidden_size)
        self.s2h = nn.Linear(feature_size,hidden_size)
        self.wz = nn.Linear(feature_size*2,1)
        self.Predictor = nn.Sequential(
            nn.Linear(hidden_size,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self, emb, structure_feature):
        emb_feature = self.emb_layers(emb)
        stc_feature = self.structured_layers(structure_feature)
        emb_hidden = torch.tanh(self.e2h(emb_feature))
        stc_hidden = torch.tanh(self.s2h(stc_feature))
        fusion_weights = torch.sigmoid(self.wz(torch.cat([emb_feature,stc_feature],dim=1)))
        fusion_feature = fusion_weights*emb_hidden * (1-fusion_weights)*stc_hidden
        out = self.Predictor(fusion_feature)
        out = out.squeeze(dim=-1)
        return out

class Latefusion_Ensemble(nn.Module):
    """
    A late fusion version of Multimodal ranking model 
    """
    def __init__(self,emb_weight=0.001):
        super(Latefusion_Ensemble,self).__init__()
        self.weight = emb_weight
        self.emb_layers = nn.Sequential(
             nn.Linear(1280, 512),
             nn.ReLU(),
             nn.Linear(512, 256),
             nn.ReLU(),
             nn.Linear(256, 128),
             nn.ReLU(),
             nn.Linear(128, 64),
             nn.ReLU(),
             nn.Linear(64, 32),
             nn.ReLU(),
             nn.Linear(32, 1)
        )
        self.structured_layers = nn.Sequential(
             nn.Linear(676, 256),
             nn.ReLU(),
             nn.Linear(256, 64),
             nn.ReLU(),
             nn.Linear(64, 16),
             nn.ReLU(),
             nn.Linear(16, 1)
        )

    def forward(self, emb, structure_feature):
        emb_out = self.emb_layers(emb)
        enc_out = self.structured_layers(structure_feature)
        output_scores = self.weight*emb_out + (1-self.weight)*enc_out
        output_scores = output_scores.squeeze(dim=-1)
        return output_scores

class Linear_Embedding(nn.Module):
    """
    A Linear embedding to convert structured data to embeddings
    """
    def __init__(self, input_size=676, emb_size=1280):
        super(Linear_Embedding,self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(1,emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
    def forward(self, x):
        bs = x.shape[0]
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n emb -> batchsize n emb', batchsize=bs)
        x = torch.cat([cls_tokens, x], dim=1)
        return x

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot ProductAttention
    """
    def __init__(self,dropout=0.1):
        super(ScaledDotProductAttention,self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,q,k,v,mask=None):
        """
        q,k,v: tensor, [b,nhead,len,dim]

        """
        scale = q.shape[-1] ** 0.5 
        attn = torch.matmul(q/scale,k.transpose(2,3))  # [b,nhead,len,len]
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.nn.functional.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    """
    Contains attention operation and Add & Norm operation
    """

    def __init__(self,d_model, d_k, d_v, n_head = 4, dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # Here, Q,K,V is input features shape:[len,n_emb]
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) #[b,len,n_head,d_k]
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k) #[b,len,n_head,d_k]
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v) #[b,len,n_head,d_v]

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [b,nhead,len,d_k]

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        #q (sz_b,len_q,n_head,N * d_k)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)
        return q, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(torch.nn.functional.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(d_model, d_k, d_v, n_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class SelfAttn(nn.Module):
    """
    Self attenion block
    """
    def __init__(self,d_model, d_inner, d_k, d_v, dropout=0.1, n_layers=6, n_head=4):
        super(SelfAttn,self).__init__()
        self.layer4stc = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        # self.layer4_pretrain = nn.ModuleList([
        #     EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        #     for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self,stc_feat,return_attns=False):
        stc_slf_attn_list = []
        stc_feat = self.layer_norm(stc_feat)
        for enc_layer in self.layer4stc:
            stc_feat, enc_slf_attn = enc_layer(stc_feat)
            stc_slf_attn_list += [enc_slf_attn] if return_attns else []

        # for enc_layer in self.layer4_pretrain:
        #     pretrain_feat, pretrain_slf_attn = enc_layer(pretrain_feat)
        #     pretrain_slf_attn_list += [pretrain_slf_attn] if return_attns else []

        # if return_attns:
        #     return stc_feat, pretrain_feat, stc_slf_attn_list, pretrain_slf_attn_list
        # return stc_feat, pretrain_feat

        if return_attns:
            return stc_feat, stc_slf_attn_list
        return stc_feat


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(d_model, d_k, d_v, n_head, dropout=dropout)
        self.enc_attn = MultiHeadAttention(d_model, d_k, d_v, n_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, 
                                        dec_input, mask=slf_attn_mask)

        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, 
                                        enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

class CrossAttn(nn.Module):
    """
    Cross attention module
    """
    def __init__(self, d_model, d_inner, d_k, d_v, dropout=0.1, n_layers=6, n_head=4):
        super(CrossAttn, self).__init__()
        self.cross_atten_layers = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self,pretrain_feat,stc_feat,return_attns=False):
        slf_attn_list, cross_attn_list = [], []
        pretrain_feat = self.layer_norm(pretrain_feat)
        for layer in self.cross_atten_layers:
            pretrain_feat, slf_attn, cross_attn = layer(pretrain_feat, stc_feat)
            slf_attn_list += [slf_attn] if return_attns else []
            cross_attn_list += [cross_attn] if return_attns else []

        if return_attns:
            return pretrain_feat, slf_attn_list, cross_attn_list
        return pretrain_feat

class TRM_FusionModel(nn.Module):
    """
    Transformer-based fusion model
    """
    def __init__(self,stc_size=676,emb_size=1280, d_inner=2048, 
                n_layers=3, n_head=4, d_k=64, d_v=64,dropout=0.1):

        super(TRM_FusionModel,self).__init__()
        self.linear_emb = Linear_Embedding(input_size=stc_size,emb_size=emb_size)
        self.slf_attn_block = SelfAttn(d_model=emb_size, d_inner=d_inner, d_k=d_k, d_v=d_v, 
                                        dropout=dropout, n_layers=n_layers, n_head=n_head)

        self.cross_attn_block = CrossAttn(d_model=emb_size, d_inner=d_inner, d_k=d_k, d_v=d_v, 
                                          dropout=dropout, n_layers=n_layers, n_head=n_head)
        self.cls_head = nn.Sequential(nn.Linear(emb_size, 512),
                                      nn.ReLU(),
                                      nn.Linear(512, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 1))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, pretrain_feat, stc_feat):
        stc_feat = self.linear_emb(stc_feat) # project structured data from [1,676,1] to [1,677,1280]
        slf_attn_output = self.slf_attn_block(stc_feat)
        cross_attn_output = self.cross_attn_block(pretrain_feat,slf_attn_output)
        # cls_feat = cross_attn_output[:,0,:]
        cls_feat = cross_attn_output.mean(1)
        output_scores = self.cls_head(cls_feat)
        output_scores = output_scores.squeeze(dim=-1)
        return output_scores

if __name__ == '__main__':
    pass
    
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


if __name__ == '__main__':
    pass
    
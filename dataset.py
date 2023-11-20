from re import A
import numpy as np
import pandas as pd
import itertools
import h5py
import torch
import torch.utils.data as data
import os
import random

class AntibactCLS_Dataset(data.Dataset):
    def __init__(self,data_file):
        self.data_df = pd.read_csv(data_file)
        # self.embeddings_dir = embeddings_dir

    def __len__(self):
        return len(self.data_df)

    def load_embeddings(self, protein_name):
        fpath = os.path.join(self.embeddings_dir,'{}.h5'.format(protein_name))
        with h5py.File(fpath, 'r') as hf:
            embeddings = hf[protein_name][:]
        return embeddings

    def __getitem__(self, idx):
        data_item = self.data_df.iloc[idx]
        label = data_item['type']
        tensor_label = torch.tensor(label, dtype=torch.float)
        # prot = self.load_embeddings(data_item['sequence'])  # embeddings, for only finetune linear
        prot =  data_item['sequence']
        res_data = {'prots':prot,'label':tensor_label}
        return res_data
        
class AntibactReg_Dateset(data.Dataset):
    """
    Dataset for regression task
    """
    def __init__(self,data_file,structured_data_dir=None):
        self.data_df = pd.read_csv(data_file)
        self.data_df= self.data_df[self.data_df['sequence'].str.len() >= 6]
        self.structured_data_dir = structured_data_dir
        
    def __len__(self):
        return len(self.data_df)

    def _get_structured_features(self, protein_name):

        fpath = os.path.join(self.structured_data_dir,'{}.h5'.format(protein_name))
        with h5py.File(fpath, 'r') as hf:
            features = hf[protein_name][:]
        return features

    def __getitem__(self, idx):
        data_item = self.data_df.iloc[idx]
        mic = data_item['MIC']
        mic = torch.tensor(mic, dtype=torch.float)
        prot =  data_item['sequence']
        enc = self._get_structured_features(prot)
        input_data = {'prots':prot,'gt':mic}
        if self.structured_data_dir:
            enc = self._get_structured_features(prot)
            input_data['enc'] = enc
        return input_data

class LambdaRank_dataset(data.Dataset):
    """
    Dataset of LambdaRank
    """
    def __init__(self, data_file,embeddings_dir,structured_dir):
        self.data_df = pd.read_csv(data_file)
        self.data_df= self.data_df[self.data_df['sequence'].str.len() >= 6]
        self.embeddings_dir = embeddings_dir
        self.structured_dir = structured_dir
        self._preprocess()

    def _preprocess(self):
        self.seq2mic = dict(zip(self.data_df['sequence'],self.data_df['MIC']))
        self.all_seqs = list(set(self.data_df['sequence']))
        random.shuffle(self.all_seqs)

    def load_embeddings(self, protein_name):
        fpath = os.path.join(self.embeddings_dir,'{}.h5'.format(protein_name))
        with h5py.File(fpath, 'r') as hf:
            embeddings = hf[protein_name][:]
        return embeddings

    def _get_structured_features(self, protein_name):
        fpath = os.path.join(self.structured_dir,'{}.h5'.format(protein_name))
        with h5py.File(fpath, 'r') as hf:
            features = hf[protein_name][:]
        return features

    def __getitem__(self, idx):
        seq = self.all_seqs[idx]
        mic = self.seq2mic[seq]
        emb = self.load_embeddings(seq)
        data = {'seq':seq,'emb':emb, 'mic':mic}
        if self.structured_dir is not None:
            enc = self._get_structured_features(seq)
            data['struct_data'] = enc

        return data

    def __len__(self):
        return len(self.all_seqs)

    def generate_weights(self):
        """
        generate weights for weighted random sampler
        """
        pos_num = len(self.data_df[self.data_df['MIC'] < 3.913])
        neg_num = len(self.data_df[self.data_df['MIC'] > 3.913])
        ratio = (neg_num / pos_num) / 4
        weights = [ratio if (self.seq2mic[seq]<3.913) else 1 for seq in self.all_seqs]
        return weights


if __name__ == "__main__":
    pass

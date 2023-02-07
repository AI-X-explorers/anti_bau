from re import A
import numpy as np
import pandas as pd
import itertools
import h5py
import torch
import torch.utils.data as data
import os
import random

# load pretrain models
import esm

class PPI_Dataset(data.Dataset):
    def __init__(self,ppi_file, embeddings_dir, maxlen = 600):

        self.maxlen = maxlen
        self.ppi_df = pd.read_csv(ppi_file)
        self.embeddings_dir = embeddings_dir

    def __len__(self):
        return len(self.ppi_df)

    def load_embeddings(self, protein_name):
        fpath = os.path.join(self.embeddings_dir,'{}.h5'.format(protein_name))
        with h5py.File(fpath, 'r') as hf:
            embeddings = hf[protein_name][:]
        return embeddings

    def __getitem__(self, idx):
        data_item = self.ppi_df.iloc[idx]
        label = data_item['label']
        tensor_label = torch.tensor(label, dtype=torch.float)
        prot_a,prot_b = self.load_embeddings(data_item['item_id_a']),self.load_embeddings(data_item['item_id_b'])
        input_data = {'prot_a':prot_a,'prot_b':prot_b,'label':tensor_label}
        return input_data

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

# class AntibactRanking_Dateset(data.Dataset):
#     """
#     For training of ranking
#     """
#     def __init__(self, data_file, embeddings_dir):
#         self.data_df = pd.read_csv(data_file)
#         self.embeddings_dir = embeddings_dir
#         self.__preprocess()

#     def __preprocess(self):
#         self.seq2mic = dict(zip(self.data_df['sequence'],self.data_df['MIC']))
#         all_seqs = set(self.data_df['sequence'])
#         self.pairs = list(itertools.product(all_seqs,all_seqs))

#     def load_embeddings(self, protein_name):
#         fpath = os.path.join(self.embeddings_dir,'{}.h5'.format(protein_name))
#         with h5py.File(fpath, 'r') as hf:
#             embeddings = hf[protein_name][:]
#         return embeddings

#     def calculate_labels(self, seq_i, seq_j):
#         """
#         Calculate ranking labels(p_ij) for seq_i, seq_j, in which
#         p_ij = 0.5*(s_ij+1)

#         s_ij = 1 if mic_i < mic_j (mic the lower,the better),
#         s_ij = 0 if mic_i == mic_j,
#         s_ij = -1 if mic_i > mic_j
#         """
#         mic_i,mic_j = self.seq2mic[seq_i],self.seq2mic[seq_j]
#         s_ij = np.sign(mic_j - mic_i)
#         p_ij = (s_ij + 1) / 2 
#         return p_ij

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, idx):
#         # seq_a,seq_b = next(self.pairs)
#         seq_a,seq_b = self.pairs[idx]
#         label = self.calculate_labels(seq_a,seq_b)
#         label = torch.tensor(label, dtype=torch.float)
#         emb_a,emb_b = self.load_embeddings(seq_a),self.load_embeddings(seq_b)
#         return {'emb_a':emb_a,'emb_b':emb_b,'label':label}

class AntibactRanking_Dateset(data.Dataset):
    """
    For training of ranking, using structured data
    """
    def __init__(self, data_file, embeddings_dir, use_dae, structured_data_dir = None, dae_encodings_dir = None):
        self.data_df = pd.read_csv(data_file)
        self.data_df= self.data_df[self.data_df['sequence'].str.len() >= 6]
        self.embeddings_dir = embeddings_dir
        self.__preprocess()
        self.use_dae = use_dae
        self.structured_data_dir = dae_encodings_dir
        if not self.use_dae:
            assert structured_data_dir is not None,"need structured data,check your path"
            self.structured_data_dir = structured_data_dir
    
    def __preprocess(self):
        self.seq2mic = dict(zip(self.data_df['sequence'],self.data_df['MIC']))
        all_seqs = set(self.data_df['sequence'])
        all_seqs = [i for i in all_seqs if len(i) >= 6]
        self.pairs = list(itertools.product(all_seqs,all_seqs))

    def load_embeddings(self, protein_name):
        fpath = os.path.join(self.embeddings_dir,'{}.h5'.format(protein_name))
        with h5py.File(fpath, 'r') as hf:
            embeddings = hf[protein_name][:]
        return embeddings

    def calculate_labels(self, seq_i, seq_j):
        """
        Calculate ranking labels(p_ij) for seq_i, seq_j, in which
        p_ij = 0.5*(s_ij+1)

        s_ij = 1 if mic_i < mic_j (mic the lower,the better),
        s_ij = 0 if mic_i == mic_j,
        s_ij = -1 if mic_i > mic_j
        """
        mic_i,mic_j = self.seq2mic[seq_i],self.seq2mic[seq_j]
        s_ij = np.sign(mic_j - mic_i)
        p_ij = (s_ij + 1) / 2 
        return p_ij

    def _get_structured_features(self, protein_name):

        fpath = os.path.join(self.structured_data_dir,'{}.h5'.format(protein_name))
        with h5py.File(fpath, 'r') as hf:
            features = hf[protein_name][:]
        return features

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seq_a,seq_b = self.pairs[idx]
        label = self.calculate_labels(seq_a,seq_b)
        label = torch.tensor(label, dtype=torch.float)
        emb_a,emb_b = self.load_embeddings(seq_a),self.load_embeddings(seq_b)
        enc_a,enc_b = self._get_structured_features(seq_a),self._get_structured_features(seq_b)
        return {'emb_a':emb_a,'emb_b':emb_b,'enc_a':enc_a,'enc_b':enc_b,'label':label}
        
class AntibactRankingTest_Dateset(data.Dataset):
    """
    A dataset for test task of antibact ranking 
    """
    def __init__(self, data_file, embeddings_dir, use_dae, structured_data_dir = None, dae_encodings_dir = None):
        self.data_df = pd.read_csv(data_file)
        self.data_df= self.data_df[self.data_df['sequence'].str.len() >= 6]
        self.embeddings_dir = embeddings_dir
        self.use_dae = use_dae
        self.structured_data_dir = dae_encodings_dir
        
        if not self.use_dae:
            assert structured_data_dir is not None,"need structured data,check your path"
            self.structured_data_dir = structured_data_dir

    def __len__(self):
        return len(self.data_df)

    def load_embeddings(self, protein_name):
        fpath = os.path.join(self.embeddings_dir,'{}.h5'.format(protein_name))
        with h5py.File(fpath, 'r') as hf:
            embeddings = hf[protein_name][:]
        return embeddings

    def _get_structured_features(self, protein_name):
        fpath = os.path.join(self.structured_data_dir,'{}.h5'.format(protein_name))
        with h5py.File(fpath, 'r') as hf:
            features = hf[protein_name][:]
        return features

    def __getitem__(self, idx):
        data_item = self.data_df.iloc[idx]
        mic = data_item['MIC']
        mic = torch.tensor(mic, dtype=torch.float)
        seq =  data_item['sequence']
        embeddings = self.load_embeddings(seq)
        enc = self._get_structured_features(seq)
        res_data = {'seq':seq,'mic':mic,'emb':embeddings,'enc':enc}
        return res_data

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

class DAE_Dateset(data.Dataset):
    """
    For denoising auto encoder
    """
    def __init__(self, data_path, structured_data_dir, mode):
        data = pd.read_csv(data_path)
        data = data['sequence'].tolist()
        data = [i for i in data if len(data) >= 6]
        split = int(len(data)*0.8)
        self.structured_data_dir = structured_data_dir

        if mode == 'train':
            self.sequence = data[:split]
        elif mode == 'test' or mode == 'val':
            self.sequence = data[split:]
        else:
            print("wrong mode, please ensure mode is one of train,test,val")
            exit()

    def __len__(self):
        return len(self.sequence)

    def _get_structured_features(self, protein_name):
        fpath = os.path.join(self.structured_data_dir,'{}.h5'.format(protein_name))
        with h5py.File(fpath, 'r') as hf:
            features = hf[protein_name][:]
        return features

    def __getitem__(self, idx):
        seq = self.sequence[idx]
        feature = self._get_structured_features(seq)
        return feature

if __name__ == "__main__":
    # embeddings_dir = 'data_antibact/final_data/pretrain_based/protein_embeddings_cls_step1'
    # train_path = '/ssd1/zhangwt/DrugAI/projects/esm/data_antibact/final_data/pretrain_based/train4prediction/ranking_finetune1_train.csv'
    # train_dataset = LambdaRank_dataset(train_path,embeddings_dir)
    # weigths = train_dataset.generate_weights()
    pass

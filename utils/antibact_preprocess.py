import numpy as np
import torch
import pandas as pd
import h5py
import os
from tqdm import tqdm
import random
import sys
sys.path.append('/ssd1/zhangwt/DrugAI/projects/esm')
import esm_project
from copy import deepcopy
from model import DenoisingAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def para_state_dict(model, ckpt_path):
    """
    load reusable parameters
    """
    state_dict = deepcopy(model.state_dict())
    assert os.path.exists(ckpt_path)
    loaded_paras = torch.load(ckpt_path)
    new_load_paras = {}
    for k,v in loaded_paras.items():
        k = k.replace('ProteinBert.','')
        new_load_paras[k] = v
    del loaded_paras

    for key in state_dict:
        if key in new_load_paras and state_dict[key].size() == new_load_paras[key].size():
            print("Success load parms: ", key)
            state_dict[key] = new_load_paras[key]
    return state_dict

class EmbeddingProcessor:
    """
    Generate esm embeddings for all proteins of all datasets
    """
    def __init__(self, prior_model = None):
        self.pretrain_model, _ = esm_project.pretrained.esm1b_t33_650M_UR50S()
        alphabet = esm_project.Alphabet.from_architecture("roberta_large")
        self.batch_converter = alphabet.get_batch_converter()
        self.pretrain_model = self.pretrain_model.to(device)
        if prior_model is not None:
            state_dict = para_state_dict(self.pretrain_model, prior_model)
            self.pretrain_model.load_state_dict(state_dict)

    def get_seqs_from_datasets(self, dataset_dir):
        datasets = [os.path.join(dataset_dir,i) for i in os.listdir(dataset_dir)]
        self.all_seqs = []
        for file in datasets:
            df = pd.read_csv(file)
            self.all_seqs.extend(df['sequence'].values)
        self.all_seqs = set(self.all_seqs)

    def get_seqs(self, seq_list):
        self.all_seqs = seq_list

    def generate_embeddings(self, out_dir, dirname = 'protein_embeddings'):
        embedding_out_dir = os.path.join(out_dir, dirname)
        os.makedirs(embedding_out_dir, exist_ok=True)
        print("Start generate embeddings, total {} ".format(len(self.all_seqs)))
        for seq in tqdm(self.all_seqs):
            f_name = os.path.join(embedding_out_dir,'{}.h5'.format(seq))
            if os.path.exists(f_name):
                continue
            data = [(seq,seq)]
            _, _, batch_tokens = self.batch_converter(data) 
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = self.pretrain_model(batch_tokens, repr_layers=[33], return_contacts=True)
                token_representations = results["representations"][33]
            pool_embedding = token_representations.mean(1).squeeze(0)  # [1280]
            pool_embedding = pool_embedding.cpu().numpy()
            # pool_embedding = token_representations.squeeze(0)
            # pool_embedding = pool_embedding.cpu().numpy()
            with h5py.File(f_name, 'w') as hf:
                hf.create_dataset(seq, data=pool_embedding)       

class encodingProcessor:
    """
    Generate encoding for structured data using denoising autoencoder
    """
    def __init__(self, ckpt_path,  norm_parms_path):
        self.model = DenoisingAE().to(device)
        # self.model.load_state_dict(torch.load(ckpt_path))
        self.norm_parms = np.load(norm_parms_path)

    def normalization(self):
        pass

    def get_data_from_dataset(self, file):

        df = pd.read_csv(file)
        newDataFrame = pd.DataFrame(index=df.index)
        columns = df.columns.tolist()
        for col_idx,col in enumerate(columns):
            if (col == 'sequence'):
                newDataFrame[col] = df[col].tolist()
            else:
                data = df[col]
                
                # col_mean = self.norm_parms[col_idx-1,0]
                # col_std = self.norm_parms[col_idx-1,1]
                # Seems that use norm,std of itself is better
                col_mean  = data.mean()
                col_std = data.std()
                if col_std != 0:
                    data = ((data - col_mean) / col_std )
                contain_nan = (True in np.isnan(data.values))
                if contain_nan:
                    print("nan col, std is {} mean is: {}".format(data.std(),data.mean()))
                newDataFrame[col] = data
                print("std is {} mean is: {}".format(data.std(),data.mean()))
        self.df = newDataFrame

    def generate_encodings(self, out_dir, dirname = 'dae_encodings'):
            encodings_out_dir = os.path.join(out_dir, dirname)
            os.makedirs(encodings_out_dir, exist_ok=True)
            print("Start generate encodings, total {} ".format(len(self.df)))

            self.model.eval()
            loss_fn = torch.nn.MSELoss().to(device)
            total_loss = 0
            for i in tqdm(range(len(self.df))):
                
                seq = self.df.iloc[i,0]
                # data = self.df.iloc[i,1:].to_records()
                data = np.array(self.df.iloc[i,1:]).astype(np.float32)
                data = torch.tensor(data)
                
                data = data.to(device)
                f_name = os.path.join(encodings_out_dir,'{}.h5'.format(seq))
                with torch.no_grad():
                    enc = self.model(data,inference=True)
                    enc = enc.cpu().numpy()
                    temp = self.model(data)
                    loss = loss_fn(temp,data)
                    total_loss += loss
                with h5py.File(f_name, 'w') as hf:
                    hf.create_dataset(seq, data=enc)    
            print("finished, avg reconstruction loss: {:.3f}".format(total_loss/len(self.df)))   

    def generate_normed_data(self, out_dir, dirname = 'normed_strutured_data'):
        """
        Just normalized the dataframe, without any process.
        """
        out_dir = os.path.join(out_dir, dirname)
        os.makedirs(out_dir, exist_ok=True)
        print("Start generate encodings, total {} ".format(len(self.df)))
        for i in tqdm(range(len(self.df))):
            seq = self.df.iloc[i,0]
            # data = self.df.iloc[i,1:].to_records()
            data = np.array(self.df.iloc[i,1:]).astype(np.float32)
            data = torch.tensor(data)
            f_name = os.path.join(out_dir,'{}.h5'.format(seq))
            with h5py.File(f_name, 'w') as hf:
                hf.create_dataset(seq, data=data) 

if __name__ == '__main__':
    """
    A example for esm embedding generation
    """
    dataset_dir = '/ssd1/zhangwt/DrugAI/projects/esm/data_antibact/final_data/pretrain_based/train4prediction'
    # dataset_dir = 'antibact_prediction/393w_bau/temp'
    # prior_model = 'antibact_final_training/cls_finetune1/step1/final.ckpt'
    prior_model = None

    embedding_processor = EmbeddingProcessor(prior_model=prior_model)
    embedding_processor.get_seqs_from_datasets(dataset_dir)
    embedding_processor.generate_embeddings(out_dir='./data_antibact/final_data/pretrain_based/',dirname='protein_embeddings_esm')
    # embedding_processor.generate_embeddings(out_dir='./antibact_prediction',dirname='protein_embeddings_full')
    """
    A example for Denosing autoencoder encoding generation
    """
    # generate for ranking,reg
    # dataset_path = './data_antibact/final_data/pretrain_based/structured_data_csv/data4training.csv'
    # model_ckpt = './ranking_exp/dae/best/final.ckpt'
    # norm_parms = 'antibact_prediction/norm_args/data4daetraining_args.npy' 
    # out_dir = './data_antibact/final_data/pretrain_based'
    # encodingProcessor = encodingProcessor(model_ckpt,norm_parms)
    # encodingProcessor.get_data_from_dataset(dataset_path)
    # encodingProcessor.generate_encodings(out_dir=out_dir)
    

    # generate dae encodings for prediction
    # dataset_path = './data_antibact/final_data/pretrain_based/structured_data_csv/data4pred_bau.csv'
    # model_ckpt = './ranking_exp/dae/best/final.ckpt'
    # norm_parms = './antibact_prediction/norm_args/data4daetraining_args.npy'
    # out_dir='./antibact_prediction'
    # encodingProcessor = encodingProcessor(model_ckpt,norm_parms)
    # encodingProcessor.get_data_from_dataset(dataset_path)
    # encodingProcessor.generate_encodings(out_dir=out_dir)

    # generate normed structured data for ranking,reg training
    # dataset_path = './data_antibact/final_data/pretrain_based/structured_data_csv/data4training.csv'
    # model_ckpt = './ranking_exp/dae/best/final.ckpt'
    # norm_parms = './antibact_prediction/norm_args/data4daetraining_args.npy'
    # out_dir = './data_antibact/final_data/pretrain_based'
    # encodingProcessor = encodingProcessor(model_ckpt,norm_parms)
    # encodingProcessor.get_data_from_dataset(dataset_path)
    # encodingProcessor.generate_normed_data(out_dir=out_dir)

    # generate normed structured data for prediction
    # dataset_path = './data_antibact/final_data/pretrain_based/structured_data_csv/data4pred_bau.csv'
    # model_ckpt = './ranking_exp/dae/best/final.ckpt'
    # norm_parms = './antibact_prediction/norm_args/data4daetraining_args.npy'
    # out_dir = './antibact_prediction'
    # encodingProcessor = encodingProcessor(model_ckpt,norm_parms)
    # encodingProcessor.get_data_from_dataset(dataset_path)
    # encodingProcessor.generate_normed_data(out_dir=out_dir)

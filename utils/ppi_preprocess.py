import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import h5py
import os
from tqdm import tqdm
import random
import torch
import json
import esm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EmbeddingProcessor:
    """
    Generate esm embeddings for all proteins of dataset
    """
    def __init__(self, prot_seq_path):
        self.pretrain_model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()
        self._get_prot_seq(prot_seq_path)
        self.pretrian_model = self.pretrain_model.to(device)

    def _get_prot_seq(self, prot_seq_path):
        with open(prot_seq_path,'r') as f:
            self.prot_seq = json.load(f)

    def generate_embeddings(self, out_dir):
        """
        save pooling embeddings of protein sequences output [1,1280]
        :param out_dir: folder your embeddings saved in: outdir/protein_embeddings
        :return:
        """
        out_dir = os.path.join(out_dir, 'protein_embeddings')
        os.makedirs(out_dir, exist_ok=True)
        for k,v in tqdm(self.prot_seq.items()):
            data = [(k,v)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = self.pretrain_model(batch_tokens, repr_layers=[33], return_contacts=True)
                token_representations = results["representations"][33]
            pool_embedding = token_representations.mean(1)  #[1,1280]
            pool_embedding = pool_embedding.cpu().numpy()
            with h5py.File(os.path.join(out_dir,'{}.h5'.format(k)), 'w') as hf:
                hf.create_dataset(k, data=pool_embedding)

class Dataset_Generator:
    """
    Generate ppi dataset. 
    """
    def __init__(self, action_data, pseq_data):
        self._preprocess(action_data)
        self.load_pseq_data(pseq_data)
        pass

    def _preprocess(self, action_data):
        all_data = pd.read_csv(action_data,sep='\t')
        all_data = all_data[all_data['mode'] == 'binding']
        all_data.drop(columns=['mode','action','is_directional','a_is_acting','score'],axis=1,inplace=True)
        all_data.drop_duplicates(inplace = True)
        self.all_data = all_data
        self.df_each_cnt = self.all_data.groupby('item_id_a').count()['item_id_b'].sort_values()

    def load_pseq_data(self, pseq_data):
        """
        Load dict{prot:seq}
        """
        self.pseq_dict = {}
        for line in open(pseq_data):
            line = line.strip().split('\t')
            self.pseq_dict[line[0]] = line[1]

    def select_protein(self, pseq_dict, min_pos=100, max_pos=100, maxlen=700):
        """
        Select proteins contains positive samples within [min_pos,maxpos] & length of sequence < maxlen
        """
        df_each_cnt = self.df_each_cnt
        protein_idx = list(df_each_cnt[(df_each_cnt >= min_pos) & (df_each_cnt <= max_pos)].index)
        protein_idx = set([x for x in protein_idx if len(pseq_dict[x]) <= maxlen])
        print("Proteins with positive samples within [{},{}], max length: {} total {} proteins are selected: \n"\
             .format(min_pos,max_pos,maxlen,len(protein_idx)))
        return protein_idx

    def get_sub_dict(self, maxlen=400):
        """
        Get subset of pseq_dict,which contains protein len < maxlen
        """
        sub_dict = {prot:seq for prot,seq in self.pseq_dict.items() if len(seq) <= maxlen}
        return sub_dict

    def create_protein_data(self, protein_idx, random_state=5,valid_size=0.1,test_size=0.2, maxlen=700, 
                        out_path=None,sample_ratio=100):
        """
        For protein a,create the dataset contains pos and neg samples
        maxlen: the max length of target protein (Protein B)
        """
        # Postive sample
        items = self.all_data[self.all_data['item_id_a'] == protein_idx]
        df_pos = pd.DataFrame(columns = ['item_id_a','item_id_b','sequence_a','sequence_b','label']) 
        target_proteins = pd.Series(items['item_id_b'].values)
        df_pos['item_id_b'] = target_proteins
        df_pos['item_id_a'] = protein_idx
        df_pos['label'] = 1
        df_pos['sequence_a'] = self.pseq_dict[protein_idx]
        seq_list = []
        for i in range(df_pos.shape[0]):
            target = target_proteins[i]
            target_seq = self.pseq_dict[target]
            seq_list.append(target_seq)
        seq_list = pd.Series(seq_list)
        df_pos['sequence_b'] = seq_list
    
        # filter protein B length < maxlen
        df_pos = df_pos[df_pos['sequence_b'].str.len() < maxlen]
        # Split train，test dataset
        data = df_pos.iloc[:,:-1]
        target = df_pos.iloc[:,-1]
        pos_train_x,pos_test_x, pos_train_y, pos_test_y = train_test_split(data,  
                                                                        target,  
                                                                        test_size = test_size,  
                                                                        random_state = random_state)  
        
        pos_train = pd.concat([pos_train_x,pos_train_y],axis=1)
        pos_test = pd.concat([pos_test_x,pos_test_y],axis=1)
        
        # Split train,valid dataset
        data = pos_train.iloc[:,:-1]
        target = pos_train.iloc[:,-1]
        valid_size = valid_size*(1-test_size)
        pos_train_x,pos_valid_x, pos_train_y, pos_valid_y = train_test_split(data,  
                                                                            target,  
                                                                            test_size = valid_size,  
                                                                            random_state = random_state)  
        pos_train = pd.concat([pos_train_x,pos_train_y],axis=1)
        pos_valid = pd.concat([pos_valid_x,pos_valid_y],axis=1)
        
        # Negative sample
        # Neg sample compute
        all_idx = set(self.all_data['item_id_a'])
        all_idx = set([x for x in all_idx if len(self.pseq_dict[x]) <= maxlen])
        target_idx = set(df_pos['item_id_b'])
        neg_idx = all_idx - target_idx
        sample_num = sample_ratio * len(target_idx)
        sample_num = min(len(neg_idx),sample_num)
        neg_prot = random.sample(neg_idx,sample_num)
        print("protein: {}, total samples: {}  pos : neg = {}".format(protein_idx,sample_num+len(target_idx),len(df_pos)/sample_num))
        
        # Create neg sample dataframe
        df_neg = pd.DataFrame(columns = ['item_id_a','item_id_b','sequence_a','sequence_b','label']) 
        df_neg['item_id_b'] = pd.Series(neg_prot)
        df_neg['item_id_a'] = protein_idx
        df_neg['label'] = 0
        df_neg['sequence_a'] = self.pseq_dict[protein_idx]
        seq_list = []
        for i in range(df_neg.shape[0]):
            target = neg_prot[i]
            target_seq = self.pseq_dict[target]
            seq_list.append(target_seq)
        seq_list = pd.Series(seq_list)
        df_neg['sequence_b'] = seq_list
        
        # Split train，test dataset
        neg_train_x,neg_test_x, neg_train_y, neg_test_y = train_test_split(df_neg.iloc[:,:-1],  
                                                                        df_neg.iloc[:,-1],  
                                                                        test_size = test_size,  
                                                                        random_state = random_state) 
        neg_train = pd.concat([neg_train_x,neg_train_y],axis=1)
        neg_test = pd.concat([neg_test_x,neg_test_y],axis=1)
        
        # Split train,valid dataset
        data = neg_train.iloc[:,:-1]
        target = neg_train.iloc[:,-1]
        valid_size = valid_size / (1-test_size-valid_size)
        neg_train_x,neg_valid_x, neg_train_y, neg_valid_y = train_test_split(data,  
                                                                            target,  
                                                                            test_size = valid_size,  
                                                                            random_state = random_state)  
        neg_train = pd.concat([neg_train_x,neg_train_y],axis=1)
        neg_valid = pd.concat([neg_valid_x,neg_valid_y],axis=1)
        
        train_data = pd.concat([pos_train,neg_train],axis = 0)
        test_data = pd.concat([pos_test,neg_test],axis = 0)
        valid_data = pd.concat([pos_valid,neg_valid],axis = 0)
        
        if out_path:
            os.makedirs(out_path,exist_ok=True)
            print(os.path.join(out_path,'{}_train.csv'.format(protein_idx)))
            train_data.to_csv(os.path.join(out_path,'{}_train.csv'.format(protein_idx)))
            test_data.to_csv(os.path.join(out_path,'{}_test.csv'.format(protein_idx)))
        
        return train_data,test_data,valid_data

    def create_dataset(self, protein_list, out_path, sample_ratio=100,maxlen_a=700,maxlen_b=700):
        """
        maxlen: the max length of target protein(Protein B)
        """
        out_path = os.path.join(out_path,'LenA{}_LenB{}'.format(maxlen_a,maxlen_b))
        os.makedirs(out_path,exist_ok=True)            
        all_train = []
        all_test = []
        all_valid = []
        for prot in tqdm(protein_list):
            train,test,valid = self.create_protein_data(prot,maxlen=maxlen_b,sample_ratio=sample_ratio, \
    #                                                   out_path=os.path.join(out_path,'single_protein'))
                                                        out_path=None)
            all_train.append(train)
            all_test.append(test)
            all_valid.append(valid)
        train_data = pd.concat(all_train,axis=0)
        test_data = pd.concat(all_test,axis=0)
        valid_data = pd.concat(all_valid,axis=0)
        train_data.to_csv(os.path.join(out_path,'train_data.csv'))
        test_data.to_csv(os.path.join(out_path,'test_data.csv'))
        valid_data.to_csv(os.path.join(out_path,'valid_data.csv'))
        maxlen = max(maxlen_a,maxlen_b)
        sub_dict = self.get_sub_dict(maxlen)
        print("all proteins:{}, select subset proteins: {} maxlen:{}".format(len(self.pseq_dict),len(sub_dict),maxlen))
        
        # Save protein-sequence dict of created dataset
        self.prot_seq_path = os.path.join(out_path,'pseq_dict_len{}.json'.format(maxlen))
        with open(self.prot_seq_path,'w') as f:
            json.dump(sub_dict,f)
            
        return train_data,test_data,valid_data

    def generate_dataset_pipeline(self, min_pos, max_pos, maxlen_a, maxlen_b, sample_ratio=100, out_path = 'ppi_dataset'):
        target_protein_list = self.select_protein(self.pseq_dict,min_pos=min_pos,max_pos=max_pos,maxlen=maxlen_a)
        train_data,test_data,valid_data = self.create_dataset(target_protein_list,out_path,sample_ratio=sample_ratio, \
                                                                maxlen_a=maxlen_a,maxlen_b=maxlen_b)
        print("trainset shape: {} testset shape:{} valid_data shape:{} ".format(train_data.shape,test_data.shape,valid_data.shape))

if __name__ == '__main__':
    # Step1: preprocess and create dataset
    action_data = './data/9606.protein.actions.all_connected.txt'
    pseq_data = './data/protein.STRING_all_connected.sequences.dictionary.tsv'
    out_dir = './data_ppi'
    dataset_generator = Dataset_Generator(action_data,pseq_data)
    dataset_generator.generate_dataset_pipeline(min_pos=99,max_pos=100,maxlen_a=400,maxlen_b=400,
                                                sample_ratio=100,out_path=out_dir)
    # Step2: Get embeddings and save as h5 file
    prot_seq_path = dataset_generator.prot_seq_path
    preprocessor = EmbeddingProcessor(prot_seq_path = prot_seq_path)
    preprocessor.generate_embeddings(out_dir=os.path.join(out_dir,'protein_embeddings'))

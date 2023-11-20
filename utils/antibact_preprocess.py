import numpy as np
import torch
import pandas as pd
import h5py
import os
from tqdm import tqdm
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class encodingProcessor:
    """
    Generate encoding for structured data.
    """
    def __init__(self, norm_parms=None, label_cols=[]):
        self.label_cols = label_cols
        self.df = None
        self.norm_parms = norm_parms if norm_parms else None
    
    def load_norm_parms(self, parms_path):
        """
        Load normalization params from pkl
        """
        with open (parms_path, "rb") as f:
            self.norm_parms = pickle.load(f)
        print(self.norm_parms)
    def get_norm_parms(self, data_csv_path, sample_ratio=0.1, save_path=None):
        """
        Get normalization parameters from data to be predicted
        Since the dataset is too large, downsampling is needed
        Args:
            data_csv_path: Path to directory of training data csvs
            save_path: Path to save normalization args, format: .pkl
        """
        dataframe = self.downsampling_datasets(data_csv_path,sample_ratio)
        self.norm_parms = {}
        columns = dataframe.columns.tolist()
        for col in columns:
            if (col == 'sequence') or (col in self.label_cols):
                continue
            else:
                data = dataframe[col]
                col_mean  = data.mean()
                col_std = data.std()
                parms = [col_mean,col_std]
                self.norm_parms[col] = parms

        if save_path:
            dir_name, fname = os.path.split(save_path)
            os.makedirs(dir_name,exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(self.norm_parms,f)
            print("norm parameters saved in: {}".format(save_path))
            
    def downsampling_datasets(self, data_dir, sample_ratio=0.1):
        """
        Since the dataset is too large, downsampling is needed
        Args:
            data_dir: directory of data to be predicted
            sample_ratio: sample_ratio
        Return: downsampled dataframe
        """
        df_list = [os.path.join(data_dir,i) for i in os.listdir(data_dir)]
        sampled_df = pd.DataFrame([])
        print("total {} dataframes to downsample".format(len(df_list)))
        for path in tqdm(df_list):
            print(path)
            df_temp = pd.read_csv(path)
            df_temp = df_temp.sample(frac=sample_ratio,random_state=42,replace=False)
            sampled_df = pd.concat([sampled_df,df_temp])
        sampled_df.reset_index(inplace=True,drop=True)
        print("sampled dataframe shape: ",sampled_df.shape)
        return sampled_df
            
    def get_data_from_dataset(self, file):
        assert self.norm_parms
        df = pd.read_csv(file)
        newDataFrame = pd.DataFrame(index=df.index)
        columns = df.columns.tolist()
        for col_idx,col in enumerate(columns):
            if (col == 'sequence'):
                newDataFrame[col] = df[col].tolist()
            else:
                data = df[col]
                col_mean = self.norm_parms[col][0]
                col_std = self.norm_parms[col][1]
                if col_std != 0:
                    data = ((data - col_mean) / col_std )
                contain_nan = (True in np.isnan(data.values))
                if contain_nan:
                    print("nan col, std is {} mean is: {}".format(data.std(),data.mean()))
                newDataFrame[col] = data

        self.df = newDataFrame

    def generate_normed_data(self, out_dir, fname = 'normed_strutured_data.h5'):
        """
        Save normalized dataframe to h5file
        """
        os.makedirs(out_dir,exist_ok=True)
        with h5py.File(os.path.join(out_dir,fname), 'w') as hf:
            for i in tqdm(range(len(self.df))):
                seq = self.df.iloc[i,0]
                data = np.array(self.df.iloc[i,1:]).astype(np.float32)
                data = torch.tensor(data)
                hf.create_dataset(seq, data=data)
        self.df = None

if __name__ == '__main__':
    Processor = encodingProcessor()
    '''
    # for prediction steps
    ## 1. get normed args from data 
    dataset_path = '/ssd1/zhangwt/DrugAI/projects/esm/prediction/8peptides/structured_data'
    save_path='data_antibact/final_data/pretrain_based/norm_params/8_peptides_bau.pkl'
    Processor.get_norm_parms(dataset_path,sample_ratio=0.1,save_path=save_path)
    ## 2. convert target datasets to normalized data and save 
    dataset_dir = '/ssd1/zhangwt/DrugAI/projects/esm/prediction/8peptides/structured_data'
    datasets = [os.path.join(dataset_dir, i) for i in os.listdir(dataset_dir)]
    # save_dir = '/ssd1/zhangwt/DrugAI/projects/esm/prediction/7peptides/normed_stc_data'
    save_dir = '/data/zhangwt/data4prediction/8peptides/structured_data'
    print("total {} datasets h5file to be generated".format(len(datasets)))
    for path in datasets:
        Processor.get_data_from_dataset(path)
        _, fname = os.path.split(path)
        fname, _ = os.path.splitext(fname)
        print("file {} start generation".format(os.path.join(save_dir,'{}.h5'.format(fname))))
        Processor.generate_normed_data(save_dir,'{}.h5'.format(fname))
'''
    Processor.load_norm_parms('data_antibact/final_data/pretrain_based/norm_params/8_peptides_bau.pkl')
    datasets = ['/ssd1/zhangwt/DrugAI/projects/esm/prediction/8peptides/structured_data/rule_2_0.csv',
                '/ssd1/zhangwt/DrugAI/projects/esm/prediction/8peptides/structured_data/rule_9_0.csv']
    save_dir = '/data/zhangwt/data4prediction/8peptides/structured_data'
    print("total {} datasets h5file to be generated".format(len(datasets)))

    for path in datasets:
        Processor.get_data_from_dataset(path)
        _, fname = os.path.split(path)
        fname, _ = os.path.splitext(fname)
        print("file {} start generation".format(os.path.join(save_dir,'{}.h5'.format(fname))))
        Processor.generate_normed_data(save_dir,'{}.h5'.format(fname))
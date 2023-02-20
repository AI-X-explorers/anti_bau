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
    
    def get_norm_parms(self, data_csv_path, save_path=None):
        """
        Get normalization parameters from training data
        Args:
            data_csv_path: Path to training data csv
            save_path: Path to save normalization args, format: .pkl
        """
        dataframe = pd.read_csv(data_csv_path)
        self.norm_parms = {}
        columns = dataframe.columns.tolist()
        for col in columns:
            if (col == 'Sequence') or (col in self.label_cols):
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

if __name__ == '__main__':
    
    # 1. get normed args from training data

    # 2. convert target datasets to normalized data and save 
    pass

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

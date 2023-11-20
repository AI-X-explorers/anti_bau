from concurrent.futures import process
from typing import Dict, List
from xxlimited import Str
import pandas as pd
import numpy as np
import h5py
import os
from tqdm import tqdm
import torch

class StcProcessor:
    """
    Convert structured datasets to h5 file
    Args:
        column name of label
    """
    def __init__(self, label_cols= List[Str]) -> None:
        self.allseqs = []
        self.norm_parms = None
        self.label_cols = label_cols

    def load_datasets(self, datasets: Dict[Str,pd.DataFrame]):
        """
        Load datasets to be processed.
        Args:
            datasets: Dict. train,test,val dataframe 
        """
        train_df = datasets['train']
        self.get_norm_parms(train_df)
        merged_df = pd.concat(datasets.values(),axis=0)
        merged_df.drop(columns=self.label_cols,inplace=True)
        merged_df.drop_duplicates(['sequence'], keep='first',inplace=True)
        columns = merged_df.columns.tolist()
        # newDataFrame will retain sequence and normed features, drop label cols
        for col in columns:
            if (col == 'sequence'):
                merged_df[col] = merged_df[col]
            else:
                data = merged_df[col]
                col_mean = self.norm_parms[col][0]
                col_std = self.norm_parms[col][1]
                if col_std != 0:
                    data = ((data - col_mean) / col_std )
                contain_nan = (True in np.isnan(data.values))
                if contain_nan:
                    print("nan col, std is {} mean is: {}".format(data.std(),data.mean()))
                merged_df[col] = data
        self.df = merged_df
        print("columns of normed dataframe: {}".format(len(self.df.columns.tolist())))

    def get_norm_parms(self, dataframe):
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
            
    def generate_normed_data(self, outdir, fname = 'normed_strutured_data.h5'):
        """
        Generate normed structured data for all sequence and save.
        Args:
            outdir: dir path to save
            fname: filename
        """
        os.makedirs(outdir,exist_ok=True)
        with h5py.File(os.path.join(outdir,fname), 'w') as hf:
            for i in tqdm(range(len(self.df))):
                seq = self.df.iloc[i,0]
                data = np.array(self.df.iloc[i,1:]).astype(np.float32)
                data = torch.tensor(data)
                hf.create_dataset(seq, data=data)

if __name__ == '__main__':
    # datadir = './datasets/stc_datasets/DeepAmPEP'
    # outdir = './datasets/stc_info'
    # processor = StcProcessor(label_cols=['Labels'])
    # train_data = pd.read_csv(os.path.join(datadir,'train.csv'))
    # test_data = pd.read_csv(os.path.join(datadir,'test.csv'))
    # val_data = pd.read_csv(os.path.join(datadir,'val.csv'))
    # datasets = {'train':train_data,'test':test_data,'val':val_data}
    # processor.load_datasets(datasets)
    # processor.generate_normed_data(outdir,'DeepAmPEP.h5')

    processor = StcProcessor(label_cols=[])
    train_data = pd.read_csv('/ssd1/zhangwt/DrugAI/projects/esm/prediction/6peptides/2023-07-04_21:26:48/stc.csv')
    datasets = {'train':train_data}
    processor.load_datasets(datasets)
    processor.generate_normed_data('/ssd1/zhangwt/DrugAI/projects/esm/prediction/6peptides/2023-07-04_21:26:48','stc.h5')


import os
from re import S
import time
import random
import argparse
import numpy as np
import pandas as pd
import h5py
import esm
from tqdm import tqdm
import torch
import torch.nn as nn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import namedtuple
from model import AntibactCLSModel,AntibactRegModel,NormalMLP
from dataset import AntibactCLS_Dataset,LambdaRank_dataset,AntibactReg_Dateset
from utils.logger import Logger
from utils.ranking_tool import ranking_tool
from utils.antibact_preprocess import EmbeddingProcessor
from utils.analyse_peptides import hydrophily_analyse

class Antibact_predictor():
    def __init__(self, args):
        self.args = args
        self.model = None
        self.logger_init()
        self.load_peptides()
        alphabet = esm.Alphabet.from_architecture("roberta_large")
        self.batch_converter = alphabet.get_batch_converter()
        self.local_rank = args.local_rank
        self.enc_dir = self.args.structured_data_dir
    
    def _pipeline(self):
        
        if self.args.ranking_results == None:
            # Cls
            if self.args.cls_pos == None:
                self.cls_predict()
                if self.args.only_cls:
                    exit()
            else:
                self.logger.info("Skip classification, load results from {} ,start ranking.".format(self.args.cls_pos))
                self.postive_peptides = self.load_results(self.args.cls_pos)
            self.ranking_predict()
        else:
            self.logger.info("Skip ranking, load results from {} ,start regression.".format(self.args.ranking_results))
            self.topK_peptides = self.load_results(self.args.ranking_results)
        self.reg_predict()
 
    def predict(self):
        self._pipeline()

    def cls_predict(self):
        
        self._get_cls_model()
        self.logger.info("Step1: Classification start !")
        self.logger.info(f"{'Peptides':^12} | {'Postive':^12} | {'Elapsed':^9}")
        t0 = time.time()
        self.postive_peptides = []
        batch_size = args.cls_bs
        total_samples_times = int((len(self.all_peptides)/batch_size)) + 1
        samples_num = 0
        for idx in range(total_samples_times):
        # for idx,peptide in enumerate(self.all_peptides):
            peptides = self.all_peptides[idx*batch_size:(idx+1)*batch_size]
            # data = [(peptide,peptide)]
            data = [(i,i) for i in peptides]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.local_rank)
            with torch.no_grad():
                logits = self.model(batch_tokens)
            preds = torch.sigmoid(logits)
            preds = preds >= 0.5
            samples_num += len(preds)
            true_index = preds.cpu().numpy()
            peptides = np.array(peptides)
            self.postive_peptides.extend(list(peptides[true_index]))
            if (samples_num % 4000 == 0 and samples_num != 0) or (idx == len(self.all_peptides) - 1):
                time_elapsed = time.time() - t0
                self.logger.info(f"{samples_num:^12} | {len(self.postive_peptides):^12} | {time_elapsed:^9.2f}")
                t0 = time.time()

        # Save postive peptides
        df = pd.DataFrame(self.postive_peptides,columns=['sequence'])
        hydrophily_analyse(df)
        df.to_csv(os.path.join(self.args.out_dir,'cls_postive.csv'),index=False)
        self.model = None

    def ranking_predict(self):
        # self._generate_embeddings()
        self._get_ranking_model()

        self.logger.info("Step2: Ranking start !")
        sequences = []
        pred_scores = []
        for peptide in tqdm(self.postive_peptides):

            if self.args.rank_model == 'AntibactRegModel':
                data = [(peptide,peptide)]
                _, _, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.local_rank)
                input = batch_tokens
            elif self.args.rank_model == 'NormalMLP':
                enc_pep = torch.tensor(self.load_encodings(peptide)).to(self.local_rank)
                input = enc_pep
            with torch.no_grad():
                pred_logits = self.model(input)
                pred_scores.append(pred_logits.item())
            sequences.append(peptide)

        pred_df = pd.DataFrame({'sequence':sequences,'pred_scores':pred_scores})
        pred_df.sort_values('pred_scores', ascending= False, inplace=True)         
        pred_df = pred_df.iloc[:self.args.topK]
        self.topK_peptides = pred_df['sequence'].tolist()
        hydrophily_analyse(pred_df)
        pred_df.to_csv(os.path.join(self.args.out_dir,'ranking_results.csv'),index=False)

    def reg_predict(self):
        self._get_reg_model()
        self.logger.info("Step3: Regression start !")
        pred_Mic = []
        seqs = []
        for peptide in tqdm(self.topK_peptides):
            if self.args.reg_model == 'AntibactRegModel':
                data = [(peptide,peptide)]
                _, _, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.local_rank)
                input = batch_tokens
            elif self.args.reg_model == 'NormalMLP':
                enc_pep = torch.tensor(self.load_encodings(peptide)).to(self.local_rank)
                input = enc_pep
            with torch.no_grad():
                logits = self.model(input)
            logits = logits.item()
            pred_Mic.append(logits)
            seqs.append(peptide)
        assert seqs == self.topK_peptides
        result_df = pd.DataFrame({'sequence':self.topK_peptides,'pred_MIC':pred_Mic})
        result_df.sort_values('pred_MIC', inplace=True)
        hydrophily_analyse(result_df)
        result_df.to_csv(os.path.join(self.args.out_dir,'reg_results.csv'),index=False)

    def logger_init(self):
        self.args.out_dir = os.path.join(self.args.out_dir,time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(self.args.out_dir)
        self.logger = Logger(self.args)
        self.logger.info("argparse:")
        for arg in vars(self.args):
            self.logger.info(str(arg).format('<20')+":\t"+str(getattr(self.args, arg)).format('<'))

    def load_peptides(self):
        """
        Load all peptides for prediction
        """
        with open(self.args.peptides_path, 'r', encoding='utf-8') as f:
            self.all_peptides = [i.strip('\n') for i in f.readlines()]
        
        self.logger.info("Load peptides successful, total {} peptides to predict".format(len(self.all_peptides)))

    def load_results(self, path):
        peptides_df = pd.read_csv(path)
        peptides = peptides_df['sequence'].tolist()
        return peptides

    def load_encodings(self, protein_name):
        fpath = os.path.join(self.enc_dir,'{}.h5'.format(protein_name))
        with h5py.File(fpath, 'r') as hf:
            embeddings = hf[protein_name][:]
        embeddings = np.expand_dims(embeddings,axis=0)
        return embeddings

    def _get_cls_model(self):
        self.model = AntibactCLSModel().to(self.local_rank)
        ckpt_path = self.args.resume_cls
        assert ckpt_path is not None,'No cls model resume'
        self.model.load_state_dict(torch.load(ckpt_path))
        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        self.model.eval()
        self.logger.info("Load cls model successfully")

    def _get_ranking_model(self):
        if self.args.rank_model == 'AntibactRegModel':
            self.model = AntibactRegModel()
        elif self.args.rank_model == 'NormalMLP':
            self.model = NormalMLP()
        self.model.to(self.args.local_rank)
        ckpt_path = self.args.resume_ranking
        assert ckpt_path is not None,'No ranking model resume'
        self.model.load_state_dict(torch.load(ckpt_path))
        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        self.model.eval()
        self.logger.info("Load ranking model successfully")

    def _get_reg_model(self):
        if self.args.rank_model == 'AntibactRegModel':
            self.model = AntibactRegModel()
        elif self.args.rank_model == 'NormalMLP':
            self.model = NormalMLP()
        self.model.to(self.args.local_rank)
        ckpt_path = self.args.resume_reg
        assert ckpt_path is not None,'No reg model resume'
        self.model.load_state_dict(torch.load(ckpt_path))
        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        self.model.eval()
        self.logger.info("Load reg model successfully")

def set_seed(seed_value=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Antibact regression pretrained-based method')
    parser.add_argument('--peptides_path', default='', help='Path to all peptides data')
    parser.add_argument('--prior_model', default='antibact_final_training/cls_finetune1/step1/final.ckpt', 
                        help='prior model to generate embeddings for ranking step')
    parser.add_argument('--cls_pos', default=None, 
                        help='Load postive peptides, and then start ranking ')     
    parser.add_argument('--ranking_results', default=None, 
                        help='Load ranking top results, then start regression ')                                            
    parser.add_argument('--resume_cls', default='antibact_final_training/cls_finetune2/step2/final.ckpt', help='path to load your cls model')
    parser.add_argument('--resume_ranking', default='/ssd1/zhangwt/DrugAI/projects/esm/lambdarank_randomsap_exp/ranking_finetune2/Normed_structured/final.ckpt', help='path to load your ranking model')
    parser.add_argument('--resume_reg', default='./antibact_final_training/reg_finetune2/NormalMLP/final.ckpt', help='path to load your reg model')
    parser.add_argument('--out_dir', default='prediction/7peptides', help='folder to save output')
    parser.add_argument('--structured_data_dir', default='antibact_prediction/normed_strutured_data', help='path to load structured data')
    parser.add_argument('--topK', default=500, type=int,help='top K peptides to rank')
    parser.add_argument('--rank_model', default='NormalMLP', 
                            help='model of choice, e.g. AntibactRegModel,NormalMLP')
    parser.add_argument('--reg_model', default='NormalMLP', 
                            help='model of choice, e.g. AntibactRegModel,NormalMLP')  
    parser.add_argument('--cls_bs', default=64, help= 'batchsize of classification model')                            
    parser.add_argument('--only_cls', default=True, 
                            help='Only need cls results') 
    # for ddp
    parser.add_argument("--local_rank", default=-1, type=int)

    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    set_seed()

    predictor = Antibact_predictor(args)
    predictor.predict()


import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import esm
import utils.ranking_metrics as rm
import torch
import torch.nn as nn
# For DDP
from collections import namedtuple
from operator import attrgetter
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data.sampler import WeightedRandomSampler
from dataset import LambdaRank_dataset
from model import AntibactRegModel,NormalMLP
from tqdm import tqdm
from copy import deepcopy
from utils.logger import Logger

parser = argparse.ArgumentParser(description='Antibact ranking pretrained-based method')
parser.add_argument('--train_path', default=None, help='train data path')
parser.add_argument('--val_path', default=None, help='validation data path')
parser.add_argument('--test_path', default=None, help='test data path')
parser.add_argument('--data_dir', default='data_antibact/final_data/pretrain_based/train4prediction', help='directory of dataset')
parser.add_argument('--embeddings_dir', default='data_antibact/final_data/pretrain_based/protein_embeddings_esm', help='directory of dataset')
parser.add_argument('--structured_data_path', default='data_antibact/final_data/pretrain_based/normed_strutured_data', help='directory of structured data')
parser.add_argument('--out_dir', default='lambdarank_randomsap_exp', help='folder to save output')
parser.add_argument('--prior_model', default=None, help='introduce prior knowledge of cls model ')
parser.add_argument('--task_name', default='ranking_finetune1', 
                    help='format task_step, e.g. ranking_finetune1,ranking_finetune2')
parser.add_argument('--model_name', default='NormalMLP', 
                    help='model of choice, e.g. AntibactRegModel,NormalMLP')
parser.add_argument('--resume', default=None, help='path to load your model')
parser.add_argument('--mode', default='train', type=str,help='train or test')
parser.add_argument('--topK', default=200, type=int,help='top K peptides to rank')
parser.add_argument('--epochs', default=200,type=str, help='epochs to train the model')
parser.add_argument('--lr', default=1e-5,type=float, help='learning rate')
parser.add_argument('--eps', default=1e-8,type=float, help='default epsilon')
parser.add_argument('--batchsize', default=32,type=int, help='batchsize')
parser.add_argument('--acc_size', default=1000,type=int, help='batchsize')

Peptide_withscore = namedtuple('Peptide_withscore',('pre_idx','mic','label'))
def set_seed(seed_value=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def para_state_dict(model, ckpt_path):
    """
    load reusable parameters
    """
    state_dict = deepcopy(model.state_dict())
    assert os.path.exists(ckpt_path)
    loaded_paras = torch.load(ckpt_path)
    for key in state_dict:
        if key in loaded_paras and state_dict[key].size() == loaded_paras[key].size():
            print("Success load parms: ", key)
            state_dict[key] = loaded_paras[key]
    return state_dict

def scores_mapping(mic_list):
    """
    Map scores of mic to ranking label
    mic_list: list of mic
    return:
        true_scores: ranking label of each mic
    """
    all_mic = list(set(mic_list))
    all_mic.sort() # small to large
    sort_idx = np.argsort(mic_list)[::-1]
    labels = [0]*len(mic_list)
    cur_label = -1
    cur_mic = 1e5
    for i, idx in enumerate(sort_idx):
        if mic_list[idx] < cur_mic:
            cur_label += 1
            cur_mic = mic_list[idx]
        labels[idx] = cur_label
    labels = list(np.log2(np.array(labels)+1))    
    return labels

def eval_parms(result1,result2,task_name):
    assert (task_name == "ranking_finetune2") or (task_name == "ranking_finetune1"),"task_name is error"

    if task_name == "ranking_finetune2":
        matrix_keys = ['top10','top20','top40']
    else:
        matrix_keys = ['top10','top50','top100','top200']
    
    idx = len(matrix_keys) - 1
    while(idx >= 0):
        if result1[matrix_keys[idx]] == result2[matrix_keys[idx]]:
            idx -= 1
            continue
        else:
            return result1[matrix_keys[idx]] >= result2[matrix_keys[idx]] 

    return False

def train(args, model, train_dataloader, val_dataloader=None):

    # create logger and out_dir

    args.out_dir = os.path.join(args.out_dir,args.task_name,
                                time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    os.makedirs(args.out_dir)
    logger = Logger(args)
    # print args:
    logger.info("argparse: \n")
    for arg in vars(args):
        logger.info(str(arg).format('<20')+":\t"+str(getattr(args, arg)).format('<'))

    # Create the optimizer, 
    # because we need to finetune transformer layers, use traditional Adam,instead of Adamw
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), \
                                lr=args.lr,eps=args.eps)

    # Total number of training steps
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)                            

    logger.info("Start training...\n")

    best_score = {"top10":0,"top20":0,"top40":0,"top50":0,"top100":0,"top200":0}
    # Training
    for epoch_i in range(args.epochs):
        model.train()
        # For each batch of training data...
        for idx,input_data in tqdm(enumerate(train_dataloader)):
            if args.model_name == 'AntibactRegModel':
                prots = input_data['seq']
                data = [(prots[i],prots[i]) for i in range(len(prots))]
                _, _, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)
                pred_scores = model(batch_tokens)
            else:
                emb = input_data['emb'].to(device)
                pred_scores = model(emb)
            pred_scores_numpy = pred_scores.data.cpu().numpy()
            gt_mic = list(input_data['mic'].numpy())
            # Map mic to labels
            true_labels = scores_mapping(gt_mic)
            # Compute lambda
            order_pairs = rm.get_pairs(true_labels)
            lambdas = rm.compute_lambda(true_labels, pred_scores_numpy, order_pairs)
            lambdas_torch = torch.Tensor(lambdas).to(device)
            # Update model parameters
            pred_scores.backward(lambdas_torch, retain_graph=True)
            # Update parameters and the learning rate
            # optimizer.step()
            # scheduler.step()
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad:
                        param.data.sub_(param.grad.data * args.lr)

        # Evaluation
            if val_dataloader and (idx % 200 == 0) and (idx != 0):
                logger.info("epoch {} start evaluation".format(epoch_i))
                model.eval()
                res, _ = evaluate(args,model,val_dataloader)
                if args.task_name == "ranking_finetune1":
                    logger.info(""" Test results: total {} peptides,ndcg: {:.3f}, top10 Precision: {:.3f}, top50 Precision: {:.3f}, top100 Precision: {:.3f}, 
                            top200 Precision:{:.3f}""".format(res['data_num'],res['ndcg'],res['top10'],res['top50'],res['top100'],res['top200']))
                else:
                    logger.info("""Test results: Total {} peptides,ndcg:{:.3f}, top10 Precision: {:.3f}, top20 Precision: {:.3f}, top40 Precision: {:.3f}, 
                            """.format(res['data_num'],res['ndcg'],res['top10'],res['top20'],res['top40']))
                logger.info("-" * 70)
                logger.info('\n')

                # save best model params
                if eval_parms(res,best_score,args.task_name):
                    best_score = res
                    model_path = os.path.join(args.out_dir,"final.ckpt")
                    torch.save(model.state_dict(), model_path)
                    if args.task_name == "ranking_finetune1":
                        logger.info(""" Current best results: ndcg: {:.3f} top10 Precision: {:.3f}, top50 Precision: {:.3f}, top100 Precision: {:.3f}, 
                                    """.format(res['ndcg'],res['top10'],res['top50'],res['top100'],res['top200']))
                    else:
                        logger.info(""" Current best results: ndcg: {:.3f} top10 Precision: {:.3f}, top20 Precision: {:.3f}, top40 Precision: {:.3f}, 
                                    """.format(res['ndcg'],res['top10'],res['top20'],res['top40']))

def topK_precision(pred,gt,k):
    pred_k = pred[:k]
    gt_k = gt[:k]
    correct = 0
    for i in pred_k:
        for j in gt_k:
            if i == j:
                correct += 1
                continue
    return correct / k

def calculate_metrics(pred,gt):
    top10_precision = topK_precision(pred,gt,10)
    top20_precision = topK_precision(pred,gt,20)
    top50_precision = topK_precision(pred,gt,50)
    top40_precision = topK_precision(pred,gt,40)
    top100_precision = topK_precision(pred,gt,100)
    top200_precision = topK_precision(pred,gt,200)
    metrics = {"top10":top10_precision,"top20":top20_precision,"top40":top40_precision,"top50":top50_precision,"top100":top100_precision,
              "top200":top200_precision}
    return metrics

def evaluate(args, model, val_dataloader):
    model.eval()
    mic_list = []
    sequences = []
    pred_scores = []
    for idx,input_data in tqdm(enumerate(val_dataloader)):
        if args.model_name == 'AntibactRegModel':
            prots = input_data['seq']
            data = [(prots[i],prots[i]) for i in range(len(prots))]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            input = batch_tokens
        else:
            prots = input_data['seq']
            emb = input_data['emb'].to(device)
            input = emb

        sequences.append(prots[0])
        mic_list.extend(list(input_data['mic'].numpy()))
        with torch.no_grad():
            pred = model(input)
            pred_scores.extend(list(pred.cpu().numpy()))

    pred_sort_index = np.argsort(pred_scores)[::-1] # scores higher to lower，mic lower to higher
    gt_sort_index = np.argsort(mic_list)            # mic lower to higher
    pred_peptides = list(np.array(sequences)[pred_sort_index]) # topk pred
    mic_gt = list(np.array(mic_list)[pred_sort_index])
    pred_scores = list(np.array(pred_scores)[pred_sort_index]) 
    gt_peptides = list(np.array(sequences)[gt_sort_index])  # topk gt
    result = calculate_metrics(pred_peptides,gt_peptides)
    result['data_num'] = len(gt_peptides)
    # calculate ndcg
    true_labels = scores_mapping(mic_list)
    pred_labels = list(np.array(true_labels)[pred_sort_index])
    ndcg = rm.ndcg(pred_labels)
    result['ndcg'] = ndcg
    pred_df = pd.DataFrame({'sequence':pred_peptides,'MIC':mic_gt,'pred_scores':pred_scores})
    pred_df.sort_values('pred_scores', ascending=False, inplace=True)
    topk = 50 if args.task_name == 'ranking_finetune2' else 200
    pred_df = pred_df.iloc[:topk]
    return result,pred_df

if __name__ == '__main__':
    args = parser.parse_args()
    # ddp init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed()
    # Set model
    if args.model_name == 'AntibactRegModel':
        model = AntibactRegModel().to(device)
    elif args.model_name == 'NormalMLP':
        args.embeddings_dir = args.structured_data_path
        model = NormalMLP().to(device)
    else:
        print('please check out your model name')    
        exit()
    ckpt_path = args.resume
    # batch_converter for data 
    alphabet = esm.Alphabet.from_architecture("roberta_large")
    batch_converter = alphabet.get_batch_converter()
    # Dataloader
    if args.mode == 'train':
        args.train_path = os.path.join(args.data_dir,"{}_train.csv".format(args.task_name))
        args.val_path = os.path.join(args.data_dir,"{}_test.csv".format(args.task_name))
        train_dataset = LambdaRank_dataset(args.train_path,args.embeddings_dir,None)
        weights = train_dataset.generate_weights()
        train_sampler = WeightedRandomSampler(weights,num_samples=args.batchsize*len(train_dataset),replacement=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, \
                                                        num_workers=0, sampler=train_sampler)
        val_dataset = LambdaRank_dataset(args.val_path,args.embeddings_dir,None)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, \
                                                    num_workers=0,shuffle=False)
        # freeze parms
        if args.model_name == 'AntibactRegModel':
            for name, value in model.named_parameters():
                if 'layers.32' in name or "Predictor" in name:
                    value.requires_grad = True
                else:
                    value.requires_grad = False 

        # load resume       
        if ckpt_path is not None:
            print("Task name: {} \nResume from: {}".format(args.task_name,args.resume))
            model.load_state_dict(torch.load(ckpt_path))

        # load prior params from cls model,is used in first training
        elif args.prior_model is not None:    
            print("Task name: {} \nLoad state dict from {}".format(args.task_name,args.prior_model))
            state_dict = para_state_dict(model, args.prior_model)
            model.load_state_dict(state_dict)
    
        train(args, model, train_dataloader, val_dataloader)

    if args.mode == 'test':
        args.test_path = os.path.join(args.data_dir,"{}_test.csv".format(args.task_name))
        test_dataset = LambdaRank_dataset(args.test_path,args.embeddings_dir)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, \
                                                    num_workers=0,shuffle=False)     
        # load model checkpoint
        if ckpt_path is not None:
            model.load_state_dict(torch.load(ckpt_path))
            print('successfully load ckpt from:{}'.format(ckpt_path))
        
        print('Start evaluation')
        res,pred_df = evaluate(args,model,test_dataloader)
        
        # Save predictions
        args.out_dir = os.path.join(args.out_dir,args.task_name)
        os.makedirs(args.out_dir,exist_ok=True)      
        pred_df.to_csv(os.path.join(args.out_dir,'{}_predictions.csv'
                       .format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))),index=False)

        if args.task_name == "ranking_finetune1":
            print(""" Test results: total {} peptides,ndcg:{:.3f}, top10 Precision: {:.3f}, top50 Precision: {:.3f}, top100 Precision: {:.3f}, 
                      top200 Precision:{:.3f}""".format(res['data_num'],res['ndcg'],res['top10'],res['top50'],res['top100'],res['top200']))
        else:
            print("""Test results: Total {} peptides,ndcg:{:.3f}, top10 Precision: {:.3f}, top20 Precision: {:.3f}, top40 Precision: {:.3f}, 
                    """.format(res['data_num'],res['ndcg'],res['top10'],res['top20'],res['top40']))
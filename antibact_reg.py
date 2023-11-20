from asyncio.log import logger
from audioop import avg
from copy import deepcopy
import os
from re import M
import time
import random
import argparse
from tkinter import N
import numpy as np
import pandas as pd
import esm
from tqdm import tqdm
import torch
import torch.nn as nn
# For DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import AntibactReg_Dateset
from model import AntibactRegModel,NormalMLP
from utils.logger import Logger

parser = argparse.ArgumentParser(description='Antibact regression pretrained-based method')
parser.add_argument('--train_path', default=None, help='train data path')
parser.add_argument('--val_path', default=None, help='validation data path')
parser.add_argument('--test_path', default=None, help='test data path')
parser.add_argument('--data_dir', default='data_antibact/final_data/pretrain_based/train4prediction', help='directory of dataset')
parser.add_argument('--structured_data_dir', default='data_antibact/final_data/pretrain_based/normed_strutured_data', help='directory of structured data')
parser.add_argument('--prior_model', default=None, help='introduce prior knowledge of cls model ')
parser.add_argument('--model_name', default='AntibactRegModel', 
                    help='model of choice, e.g. AntibactRegModel,NormalMLP')
parser.add_argument('--resume', default=None, help='path to load your model')
parser.add_argument('--out_dir', default='antibact_final_training', help='folder to save output')
parser.add_argument('--task_name', default='reg_finetune1', 
                    help='format task_step, e.g. reg_finetune1,cls_finetune2')
parser.add_argument('--mode', default='train', type=str,help='train or test')
parser.add_argument('--epochs', default=40,type=int, help='epochs to train the model')
parser.add_argument('--lr', default=1e-4,type=float, help='learning rate')
parser.add_argument('--eps', default=1e-8,type=float, help='default epsilon')
parser.add_argument('--batchsize', default=4,type=int, help='batchsize')
# for ddp
parser.add_argument("--local_rank", default=-1, type=int)

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

def eval_parms(result1,result2,task_name):
    assert (task_name == "reg_finetune2") or (task_name == "reg_finetune1"),"task_name is error"

    if task_name == "reg_finetune2":
        matrix_keys = ['top20_mse','top10_mse','top40_mse']
    else:
        matrix_keys = ['to40','top20_mse','top10_mse']
    
    idx = len(matrix_keys) - 1
    while(idx >= 0):
        if result1[matrix_keys[idx]] == result2[matrix_keys[idx]]:
            idx -= 1
            continue
        else:
            return result1[matrix_keys[idx]] < result2[matrix_keys[idx]]

    return False

def train(args, model, train_dataloader, loss_fn, val_dataloader=None):

    # create logger and out_dir
    if dist.get_rank() == 0:
        args.out_dir = os.path.join(args.out_dir,args.task_name,
                                    time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(args.out_dir)
    logger = Logger(args)
    
    # print args:
    if dist.get_rank() == 0:
        logger.info("argparse: \n")
        for arg in vars(args):
            logger.info(str(arg).format('<20')+":\t"+str(getattr(args, arg)).format('<'))

    # Create the optimizer, 
    # because we need to finetune transformer layers, use traditional Adam,instead of Adamw
    # optimizer = AdamW(model.parameters(),lr=args.lr,eps=args.eps)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), \
                                lr=args.lr,eps=args.eps)
    
    # Total number of training steps
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)
    if dist.get_rank() == 0:
        logger.info("Start training...\n")

    min_topKmse = {"top10_mse":1e7,"top20_mse":1e7,"top40_mse":1e7,"mse":1e7,"pos_mse":1e7}
    for epoch_i in range(args.epochs):
        if dist.get_rank() == 0:
            logger.info(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        train_dataloader.sampler.set_epoch(epoch_i)
        t0_epoch, t0_batch = time.time(), time.time()
        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()
        for idx,input_data in enumerate(train_dataloader):
            batch_counts += 1
            prots = input_data['prots']
            data = [(prots[i],prots[i]) for i in range(len(prots))]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(local_rank)
            gt = input_data['gt'].to(local_rank)
            optimizer.zero_grad()
            if args.model_name == 'NormalMLP':
                enc = input_data['enc']
                enc = enc.to(local_rank)
                logits = model(enc)    
            else:
                logits = model(batch_tokens)
            loss = loss_fn(logits, gt)
            batch_loss += loss.item()
            total_loss += loss.item()
            # Perform a backward pass to calculate gradients
            loss.backward()
            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()
            # Print the loss values and time elapsed for every 20 batches
            if (idx % 200 == 0 and idx != 0) or (idx == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch
                # Print training results
                if dist.get_rank() == 0:
                    logger.info(f"{epoch_i + 1:^7} | {idx:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
            
        logger.info("-" * 70)
        # Evaluation
        if val_dataloader:
            if dist.get_rank() == 0:
                logger.info("epoch {} start evaluation!\n".format(epoch_i+1))
                res = evaluate(args,model,val_dataloader)
                if args.task_name == "reg_finetune2":
                    logger.info(""" Test results: Total {} peptides, MSE: {:.3f}, top10 MSE: {:.3f}, top20 MSE: {:.3f}, top40 MSE: {:.3f}, 
                                    pos_MSE:{:.3f}""".format(len(val_dataloader),res['mse'],res['top10_mse'],res['top20_mse'],res['top40_mse'],res['pos_mse']))
                else:
                    logger.info(""" Test results: Total {} peptides, MSE: {:.3f}, top10 MSE: {:.3f}, top20 MSE: {:.3f}, top40 MSE: {:.3f}, 
                                    pos_MSE:{:.3f}""".format(len(val_dataloader),res['mse'],res['top10_mse'],res['top20_mse'],res['top40_mse'],res['pos_mse']))
                time_elapsed = time.time() - t0_epoch
                logger.info("-" * 70)
                logger.info('\n')
                # save best models
                                
                eval_indicators = res['top20_mse'] if args.task_name == "reg_finetune2" else res['top40_mse']
                indicators_name = 'top20_mse' if args.task_name == "reg_finetune2" else 'top40_mse'
                if eval_parms(res,min_topKmse,args.task_name):
                    min_topKmse = res
                    model_path = os.path.join(args.out_dir,"final.ckpt")
                    torch.save(model.module.state_dict(), model_path)
                    logger.info("Current min {}: {}".format(indicators_name,eval_indicators))


def calculate_matrix(pred,gt):
    """
    Return TopK MSE
    """
    # Sort by gt
    MAX_MIC = np.log10(8196)
    results =  [gt,pred]
    results = sorted(list(map(list, zip(*results))))
    results = list(map(list, zip(*results)))
    gt,pred = results[0],results[1]
    top10_mse = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(gt[0:10], pred[0:10])])
    top20_mse = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(gt[0:20], pred[0:20])])
    top40_mse = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(gt[0:40], pred[0:40])])
    mse = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(gt, pred)])
    pos_mse = np.mean([(actual - predicted) ** 2 
                                for actual, predicted in zip(gt, pred) 
                                if actual < MAX_MIC - 0.01])
    res_dict = {'top10_mse':top10_mse,'top20_mse':top20_mse,'top40_mse':top40_mse,'mse':mse,'pos_mse':pos_mse}
    return res_dict

def evaluate(args, model, val_dataloader):
    model.eval()
    # Tracking variables
    gt_list = []
    pred_list = []
    seqs = []
    for idx, input_data in tqdm(enumerate(val_dataloader)):
        prots = input_data['prots']
        data = [(prots[i],prots[i]) for i in range(len(prots))]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(local_rank)
        gt = input_data['gt'].to(local_rank)

        with torch.no_grad():
            if args.model_name == 'NormalMLP':
                enc = input_data['enc']
                enc = enc.to(local_rank)
                logits = model(enc)
            else:
                logits = model(batch_tokens)
        # Compute loss
        gt_list.extend(list(gt.cpu().numpy()))
        pred_list.extend(list(logits.cpu().numpy()))
        seqs.append(prots[0])

    res = calculate_matrix(pred_list,gt_list)
    res['sequence'] = seqs
    res['gt'] = gt_list
    res['pred'] = pred_list
    return res


if __name__ == '__main__':

    args = parser.parse_args()
    local_rank = args.local_rank
    # ddp init
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    set_seed()
    # Dataloader
    args.train_path = os.path.join(args.data_dir,"{}_train.csv".format(args.task_name))
    args.val_path = os.path.join(args.data_dir,"{}_test.csv".format(args.task_name))
    args.test_path = os.path.join(args.data_dir,"{}_test.csv".format(args.task_name))
    train_dataset = AntibactReg_Dateset(args.train_path,args.structured_data_dir)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, \
                                                   num_workers=0, sampler=train_sampler)

    val_dataset = AntibactReg_Dateset(args.val_path,args.structured_data_dir)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, \
                                                 num_workers=0,shuffle=False)  

    test_dataset = AntibactReg_Dateset(args.test_path,args.structured_data_dir)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, \
                                                 num_workers=0, sampler=test_sampler)                                             
    # Model and freeze parms
    # Set model
    if args.model_name == 'AntibactRegModel':
        model = AntibactRegModel().to(local_rank)
    elif args.model_name == 'NormalMLP':
        model = NormalMLP().to(local_rank)
    else:
        print('please check out your model name')    
        exit()
    # batch_converter for data 
    alphabet = esm.Alphabet.from_architecture("roberta_large")
    batch_converter = alphabet.get_batch_converter()
    ckpt_path = args.resume
    if args.mode == 'train':
        if args.model_name == 'AntibactRegModel':
            for name, value in model.named_parameters():
                if 'layers.32' in name or "Predictor" in name:
                    value.requires_grad = True
                else:
                    value.requires_grad = False

        # load model 
        # load resume
        if ckpt_path is not None:
            if dist.get_rank() == 0: 
                print("Task name: {} \nResume from: {}".format(args.task_name,args.resume))
                model.load_state_dict(torch.load(ckpt_path))
        # load prior params from cls model,is used in first training
        elif args.prior_model is not None:
            # assert args.task_name == 'reg_finetune1'
            if dist.get_rank() == 0:
                print("Task name: {} \nLoad state dict from {}".format(args.task_name,args.prior_model))
                state_dict = para_state_dict(model, args.prior_model)
                model.load_state_dict(state_dict)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        # Specify loss function
        loss_fn = nn.MSELoss(reduction='mean').to(local_rank)
        train(args, model, train_dataloader, loss_fn, val_dataloader)
    
    if args.mode == 'test':
        # load model checkpoint
        if dist.get_rank() == 0 and ckpt_path is not None:
            model.load_state_dict(torch.load(ckpt_path))
            print("Start evaluation")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        res = evaluate(args,model,test_dataloader)

        # Save predictions
        args.out_dir = os.path.join(args.out_dir,args.task_name)
        result_df = pd.DataFrame({'Sequence':res['sequence'],'label':res['gt'],'Prediction':res['pred']})
        result_df.to_csv(os.path.join(args.out_dir,'predictions.csv'),index=False)
        print("Test results: \nMSE: {:.5f}, top10_MSE: {:.5f}, top20_MSE: {:.5f}, top40_MSE:{:.5f}, pos_MSE:{} " \
                    .format(res['mse'],res['top10_mse'],res['top20_mse'],res['top40_mse'],res['pos_mse']))
import os
import time
import random
import argparse
import numpy as np
from sklearn.utils import shuffle
import esm
from tqdm import tqdm
import torch
import torch.nn as nn
# For DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset import AntibactCLS_Dataset
from model import AntibactCLSModel
from utils.logger import Logger

parser = argparse.ArgumentParser(description='Antibact classfication pretrain model method')
parser.add_argument('--train_path', default=None, help='train data path')
parser.add_argument('--val_path', default=None, help='validation data path')
parser.add_argument('--test_path', default=None, help='test data path')
parser.add_argument('--data_dir', default='data_antibact/final_data/pretrain_based/train4prediction', help='directory of dataset')
parser.add_argument('--resume', default=None, help='path to load your model')
parser.add_argument('--out_dir', default='antibact_final_training', help='folder to save output')
parser.add_argument('--task_name', default='cls_finetune1', 
                    help='format task_step, e.g. cls_finetune1,cls_finetune2')
parser.add_argument('--mode', default='train', type=str,help='train or test')
parser.add_argument('--epochs', default=40,type=str, help='epochs to train the model')
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

    best_score = 0
    for epoch_i in range(args.epochs):
        if dist.get_rank() == 0:
            logger.info(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        train_dataloader.sampler.set_epoch(epoch_i)
        t0_epoch, t0_batch = time.time(), time.time()
        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()
        # Check un_used parms
        # for n,p in model.named_parameters():
        #     if p.grad is None and p.requires_grad is True:
        #         print("Parameter not used:",n,p.shape)

        # For each batch of training data...
        for idx,input_data in enumerate(train_dataloader):
            batch_counts += 1
            prots = input_data['prots']
            data = [(prots[i],prots[i]) for i in range(len(prots))]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(local_rank)
            labels = input_data['label'].to(local_rank)
            optimizer.zero_grad()
            logits = model(batch_tokens)
            loss = loss_fn(logits, labels)
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
            if (idx % 100 == 0 and idx != 0) or (idx == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch
                # Print training results
                if dist.get_rank() == 0:
                    logger.info(f"{epoch_i + 1:^7} | {idx:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
            
        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        logger.info("-" * 70)

        # Evaluation
        if val_dataloader:
            if dist.get_rank() == 0:
                logger.info("epoch {} start evaluation!\n".format(epoch_i+1))
                res = evaluate(args,model,val_dataloader,loss_fn)
                logger.info(res)
                time_elapsed = time.time() - t0_epoch
                logger.info("epoch: {} evaluation results:\n,{}\n,time elapased:{} ".format(epoch_i + 1,res,time_elapsed))
                logger.info("-" * 70)
                logger.info('\n')

                # save best models
                if res['F1_score'] > best_score:
                    logger.info("Current best score: ",res)
                    best_score = res['F1_score']
                    model_path = os.path.join(args.out_dir,"final.ckpt")
                    torch.save(model.module.state_dict(), model_path)

def cal_confusion_matrix(y_true,y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1

    TP,FP,TN,FN = TP/len(y_true),FP/len(y_true),TN/len(y_true),FN/len(y_true)
    Acc = TP + TN
    Precision = TP / (TP+FP) if (TP+FP)!= 0 else 0
    Recall = TP / (TP + FN)
    F1_score = 2*Precision*Recall / (Precision + Recall) if (Precision + Recall) != 0 else 0
    res = {'TP':TP,'FP':FP,'TN':TN,'FN':FN,'Acc':Acc,'Precision':Precision,'Recall':Recall,'F1_score':F1_score}
    return res

def evaluate(args, model, dataloader, loss_fn):
    model.eval()
    # Tracking variables
    samples_num = 0
    val_loss = []
    y_true = []
    y_pred = []
    for idx, input_data in tqdm(enumerate(dataloader)):
        prots = input_data['prots']
        data = [(prots[i],prots[i]) for i in range(len(prots))]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(local_rank)
        labels = input_data['label'].to(local_rank)

        with torch.no_grad():
            logits = model(batch_tokens)
        # Compute loss
        loss = loss_fn(logits, labels)
        val_loss.append(loss.item())
        preds = torch.sigmoid(logits)
        preds = preds >= 0.5
        samples_num += len(preds)
        y_true.extend(list(labels.cpu().numpy()))
        y_pred.extend(list(preds.cpu().numpy()))
    print("total {} test examples ".format(len(y_pred)))
    res = cal_confusion_matrix(y_true,y_pred)
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
    train_dataset = AntibactCLS_Dataset(args.train_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, \
                                                   num_workers=0, sampler=train_sampler)

    val_dataset = AntibactCLS_Dataset(args.val_path)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, \
                                                 num_workers=0,shuffle=True)

    test_dataset = AntibactCLS_Dataset(args.test_path)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, \
                                                 num_workers=0, sampler=test_sampler)                                             
    # Model and freeze parms
    model = AntibactCLSModel().to(local_rank)

    # batch_converter for data 
    alphabet = esm.Alphabet.from_architecture("roberta_large")
    batch_converter = alphabet.get_batch_converter()
    ckpt_path = args.resume
    if args.mode == 'train':
        for name, value in model.named_parameters():
            if 'layers.32' in name or "classifier" in name:
                value.requires_grad = True
            else:
                value.requires_grad = False

        # load model
        if dist.get_rank() == 0 and ckpt_path is not None:
            model.load_state_dict(torch.load(ckpt_path))

        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        # Specify loss function
        loss_fn = nn.BCEWithLogitsLoss().to(local_rank)
        train(args, model, train_dataloader, loss_fn, val_dataloader)
    
    if args.mode == 'test':
        # load model checkpoint
        if dist.get_rank() == 0 and ckpt_path is not None:
            model.load_state_dict(torch.load(ckpt_path))
            print("Start evaluation")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        loss_fn = nn.BCEWithLogitsLoss().to(local_rank)
        res = evaluate(args,model,test_dataloader,loss_fn)
        print(res)

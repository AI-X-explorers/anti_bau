### Usage
#### Quick start
First clone codebase to your local directory
```bash
git clone git@github.com:AI-X-explorers/anti_bau.git
```
As a prerequisite, you must have PyTorch installed to use this repository,we recommend pytorch version 1.7.1.
You can create virtual enviroment by:
```
conda create -n torch1.7 python=3.8
pip install -r requirements.txt
```
### Train a classification model
```python
### step1: finetune with bacteria P.a
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node 2 --master_port 5200 antibact_cls.py --task_name cls_finetune1 --mode train
### step1:eval
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node 1 --master_port 5200 antibact_cls.py --task_name cls_finetune1 --mode test --resume ./antibact_final_training/cls_finetune1/step1/final.ckpt

### step2: finetune with bacteria B.A
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node 2 --master_port 5200 antibact_cls.py --task_name cls_finetune2 --mode train --resume ./antibact_final_training/cls_finetune1/step1/final.ckpt
### step2: test
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node 1 --master_port 5200 antibact_cls.py --task_name cls_finetune2 --mode test --resume ./antibact_final_training/cls_finetune2/step2/final.ckpt
```
### Train a ranking model
```python
### step1: finetune with bacteria P.a
CUDA_VISIBLE_DEVICES=0 python antibact_rank.py --task_name ranking_finetune1 --mode train
### step1: eval
CUDA_VISIBLE_DEVICES=0 python antibact_rank.py --task_name ranking_finetune1 --mode test --resume PATH-TO-YOUR-CKPT

### step2: finetune with bacteria B.A
CUDA_VISIBLE_DEVICES=0 python antibact_rank.py --task_name ranking_finetune2 --mode train --resume PATH-TO-YOUR-CKPT
CUDA_VISIBLE_DEVICES=0 python antibact_rank.py --task_name ranking_finetune2 --mode test --resume PATH-TO-YOUR-CKPT

```
### Train a regression model
```
### step1: finetune with bacteria P.a
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 antibact_reg.py --task_name reg_finetune1 --mode train --prior_model /ssd1/zhangwt/experiments/esm/antibact_experiments/cls_pretrain/step1/final.ckpt

### step1: eval
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node 1 antibact_reg.py --task_name reg_finetune1 --mode test --resume ./antibact_experiments/reg_finetune1/step1/final.ckpt

### step2: finetune with bacteria B.A
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 antibact_reg.py --task_name reg_finetune2 --mode train --resume ./antibact_experiments/reg_finetune1/step1/final.ckpt
### step2: eval
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node 1 antibact_reg.py --task_name reg_finetune2 --mode test --resume ./antibact_experiments/reg_finetune2/step2/final.ckpt
```

### Use pipeline to screen AMPs
```bash
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 9999 pipeline.py --peptides_path --peptides_path data/7_peptide/7_peptide_rule_0.txt
```
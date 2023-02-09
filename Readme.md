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
To be continue
### Train a ranking model
To be continue
### Train a regression model
To be continue

### Use pipeline to screen AMPs
```bash
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 9999 pipeline.py --peptides_path --peptides_path data/7_peptide/7_peptide_rule_0.txt
```
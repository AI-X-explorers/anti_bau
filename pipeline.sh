for num in 12 13 14 15 16
do
    CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node 1 --master_port 9997 pipeline.py --peptides_path ./antibact_final_training/7_peptide/7_peptide_rule_$num.txt
done
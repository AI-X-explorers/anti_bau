for num in 28
do
    CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node 1 --master_port 99998 pipeline.py --peptides_path ./antibact_final_training/8_peptide/8_peptide_rule_$num.txt
done
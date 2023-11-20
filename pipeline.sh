# for num in 0 1
# do
#     CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node 1 --master_port 999999 pipeline.py --cls_pos ./prediction/8peptides/rule_0_0/cls_postive.csv --out_dir ./prediction/8peptides/ranking_results --structured_data_path /data/zhangwt/data4prediction/8peptides/structured_data/rule_0_0.h5
# done
# Generate stc data
# for num in 28
# do 
#     python utils/generate_stc_csv_multi_process.py --start $num

# done
# 8 peptides prediction
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 99999978 pipeline.py --cls_pos ./prediction/8peptides/rule_9_2/cls_postive.csv --out_dir ./prediction/8peptides/ranking_results --structured_data_path /data/zhangwt/data4prediction/8peptides/structured_data/rule_9_2.h5


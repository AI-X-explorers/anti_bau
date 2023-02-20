import os
import pandas as pd
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='merge all prediction results')
    parser.add_argument('--prediction_dir', default='/ssd1/zhangwt/DrugAI/projects/esm/prediction/8peptides', help='Directory where the different prediction results saved')
    parser.add_argument('--out_dir', default='/ssd1/zhangwt/DrugAI/projects/esm/prediction/8peptides', help='Directory to merged result saved in')
    args = parser.parse_args()

    sub_dirs = [os.path.join(args.prediction_dir,i) for i in os.listdir(args.prediction_dir)]
    obj_file = 'cls_postive.csv'
    df_list = []
    for dir in tqdm(sub_dirs):
        df_path = os.path.join(dir,obj_file)
        if not os.path.exists(df_path):
            continue
        df_list.append(pd.read_csv(df_path))
    merged_df = pd.concat(df_list)
    print(merged_df.shape)
    out_path = os.path.join(args.out_dir,'ranking_results.csv')
    merged_df.to_csv(out_path,index=False)
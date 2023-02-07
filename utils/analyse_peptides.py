import os
import pandas as pd
import argparse
from tqdm import tqdm 

def hydrophily_encoding(peptides):
    hydrophily = ["S","T","C","Y","N","Q","D","E","K","R","H"] # 亲水性
    hydrophobe = ["G","A","V","L","I","M","F","W","P"] # 疏水性

    code = [0] * len(peptides)
    for idx,pep in enumerate(peptides):
       code[idx] = '1' if pep in hydrophily else '0'

    encode_res = ''.join(code) + '\t'
    return encode_res

def hydrophily_analyse(df):
    sequence = df['sequence'].tolist()
    hydrophily = []
    for seq in sequence:
        hydrophily.append(hydrophily_encoding(seq))
    df['hydrophily'] = hydrophily

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

def get_same_pep(pred,gt):
    common_list = []
    for i in pred:
        for j in gt:
            if i == j:
                common_list.append(i)
    return common_list

def calculate_metrics(pred,gt):
    top10_precision = topK_precision(pred,gt,10)
    top20_precision = topK_precision(pred,gt,20)
    top50_precision = topK_precision(pred,gt,50)
    top40_precision = topK_precision(pred,gt,40)
    top100_precision = topK_precision(pred,gt,100)
    top200_precision = topK_precision(pred,gt,200)
    top500_precision = topK_precision(pred,gt,500)
    matrix = {"top10":top10_precision,"top20":top20_precision,"top40":top40_precision,"top50":top50_precision,"top100":top100_precision,
              "top200":top200_precision,"top500":top500_precision}
    return matrix    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Antibact ranking pretrained-based method')
    parser.add_argument('--dir_path', default='antibact_prediction_mm/2022-08-02_00:45:04', help='directory to analyse')
    parser.add_argument('--mode', default='analyse', help='compare,analyse,distribution')
    parser.add_argument('--compare_file1', default=None, help='file1 to compare')
    parser.add_argument('--compare_file2', default=None, help='file2 to compare')
    args = parser.parse_args()
    if args.mode == 'analyse':
        files = [os.path.join(args.dir_path,i) for i in os.listdir(args.dir_path ) if '.csv' in i]
        for f in files:
            if 'analyse' in f:
                continue
            df = pd.read_csv(f)
            hydrophily_analyse(df)
            # df.to_csv(f.replace('.csv','_analyse.csv'),index=False)
            print(df.groupby(["hydrophily"], as_index=False).count())
    elif (args.mode == 'compare' or args.mode == 'distribution'):
        assert args.compare_file1 and args.compare_file2 is not None
        df1 = pd.read_csv(args.compare_file1)
        df2 = pd.read_csv(args.compare_file2)
        sequence1 = df1['sequence'].tolist()
        sequence2 = df2['sequence'].tolist()
        if args.mode == 'compare':
            result = calculate_metrics(sequence1,sequence2)
            com_list = get_same_pep(sequence1,sequence2)
            print(result,'\n',com_list)
        else:
            idx_list = []
            for seq in tqdm(sequence1):
                idx_list.append(sequence2.index(seq))
            df1['ranking of other'] = idx_list
            df1.to_csv('./ranking_distribution.csv',index=False)    
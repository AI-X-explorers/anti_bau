import os
import pandas as pd
import sys
from structure_data_generate.cal_pep_des import cal_pep_fromlist
from structure_data_generate import BasicDes, Autocorrelation, CTD, PseudoAAC, AAComposition, QuasiSequenceOrder
import multiprocessing
import sys
import gc
import argparse
import math
from tqdm import tqdm

def calculate_descriptors(data):
    peptides_descriptors=[]
    sequence = data["sequence"]
    peptides = sequence.values.copy().tolist()
    count = 0
    for peptide_list in tqdm(peptides):
        peptides_descriptor={}
        peptide = str(peptide_list)
        if peptide!="SSQRMW" and peptide!="WMRQSS":
            AAC = AAComposition.CalculateAAComposition(peptide)
            DIP = AAComposition.CalculateDipeptideComposition(peptide)
            MBA = Autocorrelation.CalculateNormalizedMoreauBrotoAutoTotal(peptide, lamba=5)
            CCTD = CTD.CalculateCTD(peptide)
            QSO = QuasiSequenceOrder.GetSequenceOrderCouplingNumberTotal(peptide, maxlag=5)
            PAAC = PseudoAAC._GetPseudoAAC(peptide,lamda=5)
            APAAC = PseudoAAC.GetAPseudoAAC(peptide, lamda=5)
            Basic = BasicDes.cal_discriptors(peptide)
            peptides_descriptor.update(AAC)
            peptides_descriptor.update(DIP)
            peptides_descriptor.update(MBA)
            peptides_descriptor.update(CCTD)
            peptides_descriptor.update(QSO)
            peptides_descriptor.update(PAAC)
            peptides_descriptor.update(APAAC)
            peptides_descriptor.update(Basic)
            peptides_descriptors.append(peptides_descriptor)
            count += 1
    return sequence,peptides_descriptors
    
def doit(ind):
    file = '/ssd1/zhangwt/DrugAI/projects/esm/prediction/8peptides/rule_{}/cls_postive.csv'.format(ind) # 生成的序列信息
    if os.path.exists(file):
        print(file)
        data = pd.read_csv(file,encoding="utf-8")
        print("total {} peptides to generate".format(len(data)))
        max_length = 800000
        split_num = int(math.ceil(len(data)/max_length))
        for split in range(split_num):
            if split == 1:
                exit()
            df = data[split*max_length:(split+1)*max_length]
            sequence,peptides_descriptors = calculate_descriptors(df)
            # 保存生成的结构化数据
            outdir = '/ssd1/zhangwt/DrugAI/projects/esm/prediction/8peptides/structured_data/'
            os.makedirs(outdir,exist_ok=True)
            save_path = outdir + 'rule_' + str(ind) +'_' + str(split) + '.csv'
            writeDataToExcleFile(sequence, peptides_descriptors, save_path)
            print("results csv saved in :{}".format(save_path))
            sequence,peptides_descriptors = [],[]
            gc.collect()

def writeDataToExcleFile(sequence,inputData,outPutFile):
    df = pd.DataFrame(inputData)
    sequence = sequence.reset_index(drop=True)
    print(sequence)
    result = pd.concat([sequence,df], axis=1)
    print(result)
    # result.to_csv(outPutFile,index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Antibact ranking pretrained-based method')
    parser.add_argument('--start', type=int, default=9)
    args = parser.parse_args()
    pros = []
    for i in range(args.start,args.start+1):
        process = multiprocessing.Process(target=doit, args=(i,))
        pros.append(process)
        process.start()

    for process in pros:
        process.join()
    # # generate structured data
    # data_dir = '/home/zhangwt/DrugAI/planB/AMPsMultimodalBenchmark/datasets/ori_datasets/cls_imbalanced_analyse'
    # files = os.listdir(data_dir)
    # out_dir = '/home/zhangwt/DrugAI/planB/AMPsMultimodalBenchmark/datasets/stc_datasets/cls_imbalanced_analyse'
    # os.makedirs(out_dir,exist_ok=True)
    # for file in files:
    #     data_file = os.path.join(data_dir,file)
    #     data = pd.read_csv(data_file, encoding="utf-8")  
    #     sequence = data['Sequence']
    #     labels = data['Labels']
    #     # labels = data['MIC']
    #     peptides_list = sequence.values.copy().tolist()
    #     out_path = os.path.join(out_dir,file)
    #     print("output path: {}".format(out_path))
    #     cal_pep_fromlist(peptides_list,output_path = out_path, labels=labels)

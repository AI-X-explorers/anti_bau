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
    
def doit(fpath):
    save_dir, _ = os.path.split(fpath)
    file = fpath
    if os.path.exists(file):
        print(file)
        data = pd.read_csv(file,encoding="utf-8")
        print("total {} peptides to generate".format(len(data)))
        sequence,peptides_descriptors = calculate_descriptors(data)
        # 保存生成的结构化数据
        save_path = os.path.join(save_dir,'stc.csv')
        writeDataToExcleFile(sequence, peptides_descriptors, save_path)
        print("results csv saved in :{}".format(save_path))
        gc.collect()

def writeDataToExcleFile(sequence,inputData,outPutFile):
    df = pd.DataFrame(inputData)
    sequence = sequence.reset_index(drop=True)
    result = pd.concat([sequence,df], axis=1)
    result.to_csv(outPutFile,index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate structured data csv file')
    parser.add_argument('--fpath', type=str, default='/ssd1/zhangwt/DrugAI/projects/esm/prediction/6peptides/2023-07-04_21:26:48/cls_postive.csv')
    args = parser.parse_args()
    pros = []

    process = multiprocessing.Process(target=doit, args=(args.fpath,))
    pros.append(process)
    process.start()

    for process in pros:
        process.join()
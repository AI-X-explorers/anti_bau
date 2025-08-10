import os
import pandas as pd
import sys
from structure_data_generate.cal_pep_des import cal_pep_fromlist
from structure_data_generate import BasicDes, Autocorrelation, CTD, PseudoAAC, AAComposition, QuasiSequenceOrder
from multiprocessing import Pool
import sys
import gc
import argparse
import math
from tqdm import tqdm

def calculate_descriptors(data):
    """Calculate various descriptors for peptide sequences"""
    peptides_descriptors = []
    sequence = data["sequence"]
    peptides = sequence.values.copy().tolist()
    count = 0
    
    for peptide_list in tqdm(peptides):
        peptides_descriptor = {}
        peptide = str(peptide_list)
        
        if peptide != "SSQRMW" and peptide != "WMRQSS":
            # Calculate different types of descriptors
            AAC = AAComposition.CalculateAAComposition(peptide)
            DIP = AAComposition.CalculateDipeptideComposition(peptide)
            MBA = Autocorrelation.CalculateNormalizedMoreauBrotoAutoTotal(peptide, lamba=5)
            CCTD = CTD.CalculateCTD(peptide)
            QSO = QuasiSequenceOrder.GetSequenceOrderCouplingNumberTotal(peptide, maxlag=5)
            PAAC = PseudoAAC._GetPseudoAAC(peptide, lamda=5)
            APAAC = PseudoAAC.GetAPseudoAAC(peptide, lamda=5)
            Basic = BasicDes.cal_discriptors(peptide)
            
            # Combine all descriptors
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
            
    return sequence, peptides_descriptors
    
def process_file(fpath):
    """Process a single file and generate structured data"""
    save_dir, _ = os.path.split(fpath)
    file = fpath
    
    if os.path.exists(file):
        print(file)
        data = pd.read_csv(file, encoding="utf-8")
        print("Total {} peptides to generate".format(len(data)))
        
        sequence, peptides_descriptors = calculate_descriptors(data)
        
        # Save generated structured data
        save_path = os.path.join(save_dir, 'structured_data.csv')
        save_data_to_csv(sequence, peptides_descriptors, save_path)
        print("Results CSV saved in: {}".format(save_path))
        gc.collect()

def save_data_to_csv(sequence, input_data, output_file):
    """Save data to CSV file"""
    df = pd.DataFrame(input_data)
    sequence = sequence.reset_index(drop=True)
    result = pd.concat([sequence, df], axis=1)
    result.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate structured data CSV file')
    parser.add_argument('--num_processes', type=int, default=64, help='Number of processes to use')
    parser.add_argument('--fpath', type=str, default='path/to/your/input/file.csv', 
                        help='Path to input CSV file')
    args = parser.parse_args()
    process_file(args.fpath)

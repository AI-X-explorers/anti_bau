import os
import pandas as pd
import numpy as np
import h5py
import torch
import gc
import argparse
from typing import Dict, List
from tqdm import tqdm
from structure_data_generate.cal_pep_des import cal_pep_fromlist
from structure_data_generate import BasicDes, Autocorrelation, CTD, PseudoAAC, AAComposition, QuasiSequenceOrder

class PeptideStructuralProcessor:
    """
    Complete pipeline for processing peptide sequences:
    1. Calculate structural descriptors
    2. Normalize features
    3. Save as H5 format
    """
    
    def __init__(self, label_cols: List[str] = None):
        """
        Initialize processor
        Args:
            label_cols: List of column names to be treated as labels
        """
        self.label_cols = label_cols or []
        self.norm_params = None
        self.processed_df = None
    
    def calculate_descriptors(self, data: pd.DataFrame):
        """
        Calculate various descriptors for peptide sequences
        Args:
            data: DataFrame containing 'sequence' column
        Returns:
            sequence: Series of sequences
            peptides_descriptors: List of descriptor dictionaries
        """
        peptides_descriptors = []
        sequence = data["sequence"]
        peptides = sequence.values.copy().tolist()
        
        print(f"Calculating descriptors for {len(peptides)} peptides...")
        
        for peptide_list in tqdm(peptides, desc="Processing peptides"):
            peptides_descriptor = {}
            peptide = str(peptide_list)
            
            # Skip specific problematic sequences
            if peptide not in ["SSQRMW", "WMRQSS"]:
                try:
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
                    
                except Exception as e:
                    print(f"Error processing peptide {peptide}: {e}")
                    continue
                    
            peptides_descriptors.append(peptides_descriptor)
                    
        return sequence, peptides_descriptors
    
    def save_structured_data_to_csv(self, sequence, input_data, output_file):
        """
        Save structured data to CSV file
        Args:
            sequence: Series of sequences
            input_data: List of descriptor dictionaries
            output_file: Output CSV file path
        """
        df = pd.DataFrame(input_data)
        sequence = sequence.reset_index(drop=True)
        result = pd.concat([sequence, df], axis=1)
        result.to_csv(output_file, index=False)
        print(f"Structured data saved to: {output_file}")
        return result
    
    def get_normalization_parameters(self, dataframe: pd.DataFrame):
        """
        Calculate normalization parameters from training data
        Args:
            dataframe: Training DataFrame
        """
        self.norm_params = {}
        columns = dataframe.columns.tolist()
        
        for col in columns:
            if col == 'sequence' or col in self.label_cols:
                continue
            else:
                data = dataframe[col]
                col_mean = data.mean()
                col_std = data.std()
                self.norm_params[col] = [col_mean, col_std]
    
    def normalize_and_prepare_data(self, datasets: Dict[str, pd.DataFrame]):
        """
        Normalize datasets and prepare for H5 conversion
        Args:
            datasets: Dictionary of DataFrames (e.g., {'train': df, 'test': df})
        """
        # Use training data to calculate normalization parameters
        if 'train' in datasets:
            train_df = datasets['train']
            self.get_normalization_parameters(train_df)
        else:
            # If no train set specified, use first dataset
            first_key = list(datasets.keys())[0]
            self.get_normalization_parameters(datasets[first_key])
        
        # Merge all datasets
        merged_df = pd.concat(datasets.values(), axis=0)
        
        # Remove label columns and duplicates
        if self.label_cols:
            merged_df.drop(columns=self.label_cols, inplace=True, errors='ignore')
        merged_df.drop_duplicates(['sequence'], keep='first', inplace=True)
        
        columns = merged_df.columns.tolist()
        
        # Normalize features
        print("Normalizing features...")
        for col in tqdm(columns, desc="Normalizing columns"):
            if col == 'sequence':
                continue
            else:
                data = merged_df[col]
                if col in self.norm_params:
                    col_mean = self.norm_params[col][0]
                    col_std = self.norm_params[col][1]
                    
                    if col_std != 0:
                        data = (data - col_mean) / col_std
                    
                    # Check for NaN values
                    if np.isnan(data.values).any():
                        print(f"Warning: NaN values found in column {col}")
                        print(f"Column std: {data.std()}, mean: {data.mean()}")
                
                merged_df[col] = data
        
        self.processed_df = merged_df
        print(f"Normalized dataframe has {len(self.processed_df.columns)} columns")
    
    def generate_h5_file(self, output_dir: str, filename: str = 'structured_data.h5'):
        """
        Generate H5 file containing all normalized structured data
        Args:
            output_dir: Directory to save H5 file
            filename: Name of H5 file
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        print(f"Generating H5 file: {output_path}")
        with h5py.File(output_path, 'w') as hf:
            for i in tqdm(range(len(self.processed_df)), desc="Writing to H5"):
                seq = self.processed_df.iloc[i, 0]  # sequence column
                data = np.array(self.processed_df.iloc[i, 1:]).astype(np.float32)
                hf.create_dataset(seq, data=data)
        
        print(f"H5 file saved successfully: {output_path}")
    
    def generate_individual_h5_files(self, output_dir: str):
        """
        Generate individual H5 files for each sequence
        Args:
            output_dir: Directory to save H5 files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating individual H5 files...")
        for i in tqdm(range(len(self.processed_df)), desc="Creating individual H5 files"):
            seq = self.processed_df.iloc[i, 0]
            data = np.array(self.processed_df.iloc[i, 1:]).astype(np.float32)
            
            filename = f"{seq}.h5"
            filepath = os.path.join(output_dir, filename)
            
            with h5py.File(filepath, 'w') as hf:
                hf.create_dataset(seq, data=data)
        
        print(f"Individual H5 files saved in: {output_dir}")
    
    def process_complete_pipeline(self, 
                                input_csv: str, 
                                output_dir: str, 
                                h5_filename: str = 'structured_data.h5',
                                save_csv: bool = True,
                                individual_h5: bool = False):
        """
        Complete processing pipeline from CSV to H5
        Args:
            input_csv: Path to input CSV file with 'sequence' column
            output_dir: Directory to save outputs
            h5_filename: Name of output H5 file
            save_csv: Whether to save intermediate CSV file
            individual_h5: Whether to generate individual H5 files
        """
        print("=== Starting Complete Peptide Processing Pipeline ===")
        
        # Step 1: Load input data
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input file not found: {input_csv}")
        
        data = pd.read_csv(input_csv, encoding="utf-8")
        print(f"Loaded {len(data)} sequences from {input_csv}")
        
        # Step 2: Calculate descriptors
        sequence, peptides_descriptors = self.calculate_descriptors(data)
        
        # Step 3: Save structured data to CSV (optional)
        if save_csv:
            csv_output = os.path.join(output_dir, 'structured_data.csv')
            os.makedirs(output_dir, exist_ok=True)
            structured_df = self.save_structured_data_to_csv(sequence, peptides_descriptors, csv_output)
        else:
            df = pd.DataFrame(peptides_descriptors)
            sequence = sequence.reset_index(drop=True)
            structured_df = pd.concat([sequence, df], axis=1)
        
        # Step 4: Normalize and prepare data
        datasets = {'train': structured_df}
        self.normalize_and_prepare_data(datasets)
        
        # Step 5: Generate H5 file
        self.generate_h5_file(output_dir, h5_filename)
        
        # Step 6: Generate individual H5 files (optional)
        if individual_h5:
            individual_dir = os.path.join(output_dir, 'individual_h5')
            self.generate_individual_h5_files(individual_dir)
        
        print("=== Pipeline completed successfully ===")
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description='Complete peptide processing pipeline: CSV -> Descriptors -> H5')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--h5_filename', type=str, default='structured_data.h5', help='Name of output H5 file')
    parser.add_argument('--label_cols', type=str, nargs='*', default=[], help='Label column names to exclude')
    parser.add_argument('--save_csv', action='store_true', help='Save intermediate CSV file')
    parser.add_argument('--individual_h5', action='store_true', help='Generate individual H5 files')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = PeptideStructuralProcessor(label_cols=args.label_cols)
    
    # Run complete pipeline
    processor.process_complete_pipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        h5_filename=args.h5_filename,
        save_csv=args.save_csv,
        individual_h5=args.individual_h5
    )

if __name__ == "__main__":
    # Example usage (commented out for command line usage)
    """
    processor = PeptideStructuralProcessor(label_cols=[])
    processor.process_complete_pipeline(
        input_csv='path/to/your/input.csv',
        output_dir='path/to/output/directory',
        h5_filename='peptide_data.h5',
        save_csv=True,
        individual_h5=True
    )
    """
    
    main()

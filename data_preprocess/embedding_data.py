
import pandas as pd
import numpy as np
import torch
import esm
import h5py
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class ProteinEmbeddingProcessor:
    def __init__(self, embeddings_dir='embeddings', device='cuda'):
        self.embeddings_dir = embeddings_dir
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Create directory for saving embeddings
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Load ESM model
        print(f"Loading ESM model on {self.device}...")
        self.ProteinBert, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.ProteinBert = self.ProteinBert.to(self.device)
        self.ProteinBert.eval()
        
        # Initialize batch converter
        self.batch_converter = self.alphabet.get_batch_converter()
    
    def process_csv(self, csv_file):
        """Process CSV file, add type column and provide statistics"""
        print("Processing CSV file...")
        df = pd.read_csv(csv_file)
        
        # Create type column: label < 7 -> 1, label >= 7 -> 0
        df['type'] = (df['label'] < 7).astype(int)
        
        # Count samples with type=1
        type1_count = (df['type'] == 1).sum()
        print(f"Number of samples with type=1: {type1_count}")
        print(f"Number of samples with type=0: {(df['type'] == 0).sum()}")
        
        return df
    
    def split_and_save_data(self, df, train_file='train.csv', test_file='test.csv'):
        """Split dataset into train/test (8:2) and save"""
        print("Splitting dataset into train and test...")
        
        # Split randomly with 8:2 ratio
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['type'])
        
        # Save files
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        print(f"Train set saved to {train_file}: {len(train_df)} samples")
        print(f"Test set saved to {test_file}: {len(test_df)} samples")
        
        return train_df, test_df
    
    def compute_embeddings(self, sequences, protein_names=None, batch_size=1):
        """Compute embeddings for protein sequences"""
        print(f"Computing embeddings for {len(sequences)} sequences...")
        
        if protein_names is None:
            protein_names = [f"protein_{i}" for i in range(len(sequences))]
        
        embeddings_dict = {}
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size)):
                batch_sequences = sequences[i:i+batch_size]
                batch_names = protein_names[i:i+batch_size]
                
                # Prepare data
                data = [(name, seq) for name, seq in zip(batch_names, batch_sequences)]
                
                # Convert to tokens
                batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.device)
                
                # Get model output
                results = self.ProteinBert(batch_tokens, repr_layers=[33], return_contacts=True)
                token_representations = results["representations"][33]
                
                # Get CLS token embedding
                cls_embeddings = token_representations[:, 0, :]  # CLS token
                
                # Save embedding for each sequence
                for j, (name, embedding) in enumerate(zip(batch_names, cls_embeddings)):
                    embeddings_dict[name] = embedding.cpu().numpy()
                    
                    # Save to h5 file
                    fpath = os.path.join(self.embeddings_dir, f'{name}.h5')
                    with h5py.File(fpath, 'w') as hf:
                        hf.create_dataset(name, data=embedding.cpu().numpy())
        
        return embeddings_dict
    
    def load_embedding(self, protein_name):
        """Load embedding from h5 file"""
        fpath = os.path.join(self.embeddings_dir, f'{protein_name}.h5')
        with h5py.File(fpath, 'r') as hf:
            embeddings = hf[protein_name][:]
        return embeddings
    
    def process_dataset(self, df, dataset_name='dataset'):
        """Process entire dataset"""
        sequences = df['Sequence'].tolist()
        protein_names = [f"{dataset_name}_protein_{i}" for i in range(len(sequences))]
        
        # Compute embeddings
        embeddings_dict = self.compute_embeddings(sequences, protein_names, batch_size=256)
        
        # Add embedding information to dataframe
        df['protein_name'] = protein_names
        df['embedding_path'] = [os.path.join(self.embeddings_dir, f'{name}.h5') for name in protein_names]
        
        return df, embeddings_dict

def main():
    # Initialize processor
    processor = ProteinEmbeddingProcessor(embeddings_dir='protein_embeddings_esm')
    
    # Process CSV file
    csv_file = 'path/to/your/dataset.csv'  # Replace with your CSV file path
    df = processor.process_csv(csv_file)
    
    # Split and save train/test sets
    train_df, test_df = processor.split_and_save_data(df)
    
    # Compute embeddings for training set
    print("\nProcessing training set...")
    train_df, train_embeddings = processor.process_dataset(train_df, 'train')
    train_df.to_csv('train_with_embeddings.csv', index=False)
    
    # Compute embeddings for test set
    print("\nProcessing test set...")
    test_df, test_embeddings = processor.process_dataset(test_df, 'test')
    test_df.to_csv('test_with_embeddings.csv', index=False)
    
    print("\nProcessing completed!")
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Embeddings saved in: {processor.embeddings_dir}/")
    
    # Example: load one embedding
    if len(train_df) > 0:
        sample_protein = train_df.iloc[0]['protein_name']
        sample_embedding = processor.load_embedding(sample_protein)
        print(f"\nSample embedding shape: {sample_embedding.shape}")

if __name__ == "__main__":
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name()}")
    
    main()

# Aim of this script is to get the embeddings of flanking regions based on ESM-2
# Concatenate embeddings for start and end of each peptide/propeptide
# Get the average of concatenated embeddings for each protein

import torch
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
import esm

class ESM2Embedder:
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model, self.alphabet = self._init_model()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.embed_dim = self._discover_embedding_dim()
        self.at_embedding = torch.randn(self.embed_dim, device=self.device)
    
    def _init_model(self):
        """Initialize ESM-2 model and alphabet"""
        model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
        model = model.eval().to(self.device)
        return model, alphabet
    
    def _discover_embedding_dim(self):
        """Dynamically discover embedding dimension by processing a single AA"""
        with torch.no_grad():
            batch = [("dummy", "A")]
            _, _, tokens = self.batch_converter(batch)
            tokens = tokens.to(self.device)
            results = self.model(tokens, repr_layers=[33], return_contacts=False)
            return results["representations"][33].shape[-1]  # Returns embedding dimension
    
    def embed_sequence(self, sequence: str) -> torch.Tensor:
        """Generate 2*embed_dim embedding for any input sequence"""
        if pd.isna(sequence) or not sequence:
            return torch.zeros(self.embed_dim * 2, device=self.device)
        
        # Handle special @ cases
        if sequence == '@':
            return torch.cat([self.at_embedding, self.at_embedding])
        elif sequence == '@@':
            return torch.cat([self.at_embedding, self.at_embedding])
        elif '@' in sequence:
            parts = []
            for char in sequence:
                if char == '@':
                    parts.append(self.at_embedding)
                else:
                    parts.append(self.embed_single_aa(char))
            return torch.cat(parts)
        
        # Normal AA processing
        if len(sequence) == 1:
            return torch.cat([self.embed_single_aa(sequence), self.embed_single_aa(sequence)])
        else:
            return self.embed_aa_pair(sequence)
    
    def embed_single_aa(self, aa: str) -> torch.Tensor:
        """Embed single amino acid using ESM-2"""
        batch = [("dummy", aa)]
        _, _, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)
        
        with torch.no_grad():
            results = self.model(tokens, repr_layers=[33], return_contacts=False)
        
        # Remove BOS/EOS tokens and average
        token_representations = results["representations"][33]
        return token_representations[0, 1:-1].mean(dim=0)  # [embed_dim]
    
    def embed_aa_pair(self, pair: str) -> torch.Tensor:
        """Embed AA pair and return 2*embed_dim vector using ESM-2"""
        batch = [("dummy", pair)]
        _, _, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)
        
        with torch.no_grad():
            results = self.model(tokens, repr_layers=[33], return_contacts=False)
        
        token_representations = results["representations"][33]
        # Get embeddings for each AA (removing BOS/EOS)
        aa1_emb = token_representations[0, 1]  # First AA
        aa2_emb = token_representations[0, 2]  # Second AA
        return torch.cat([aa1_emb, aa2_emb])

def process_with_esm2(csv_path: str, output_dir: str, device: str = 'cpu'):
    embedder = ESM2Embedder(device)
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    protein_embeddings = defaultdict(list)
    expected_size = embedder.embed_dim * 4  # 4*embed_dim for pre+post

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        protein_id = row['protein_id']
        coords = row['coordinates'].replace('-', '_')
        
        # Get pre and post embeddings (each 2*embed_dim)
        pre_emb = embedder.embed_sequence(row['pre-region'])
        post_emb = embedder.embed_sequence(row['post-region'])
        
        # Combine to 4*embed_dim
        combined = torch.cat([pre_emb, post_emb])
        assert combined.shape[0] == expected_size, \
               f"Size mismatch: {combined.shape[0]} != {expected_size}"
        
        # Save individual
        torch.save(combined, os.path.join(output_dir, f"{protein_id}_{coords}.pt"))
        protein_embeddings[protein_id].append(combined)
    
    # Save mean embeddings
    mean_dir = os.path.join(output_dir, "mean_embeddings")
    os.makedirs(mean_dir, exist_ok=True)
    
    for prot_id, emb_list in tqdm(protein_embeddings.items(), desc="Averaging"):
        mean_emb = torch.stack(emb_list).mean(dim=0)
        torch.save(mean_emb, os.path.join(mean_dir, f"{prot_id}_mean.pt"))

if __name__ == "__main__":
    config = {
        "csv_path": "data/flanking_regions_no_missing.csv",
        "output_dir": "data/esm2_processed_embeddings_2d",
        "device": "cuda:1" if torch.cuda.is_available() else "cpu"
    }
    process_with_esm2(**config)
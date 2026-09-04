# Aim of this script is to get the embeddings of flanking regions based on ESM-3
# Concatenate embeddings for start and end of each peptide/propeptide
# Get the average of concatenated embeddings for each protein

import torch
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from esm.pretrained import ESM3_sm_open_v0
from esm.tokenization import get_esm3_model_tokenizers

class ESM3Embedder:
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model, self.tokenizers = self._init_model()
        self.embed_dim = self._discover_embedding_dim()
        self.at_embedding = torch.randn(self.embed_dim, device=self.device)
    
    def _init_model(self):
        """Initialize ESM-3 model and tokenizers"""
        model = ESM3_sm_open_v0().eval().to(self.device)
        tokenizers = get_esm3_model_tokenizers()
        return model, tokenizers
    
    def _discover_embedding_dim(self):
        """Dynamically discover embedding dimension by processing a single AA"""
        with torch.no_grad():
            tokens = self.tokenizers.sequence.encode("A")
            token_tensor = torch.tensor(tokens).unsqueeze(0).to(self.device)
            outputs = self.model(sequence_tokens=token_tensor)
            return outputs.embeddings.shape[-1]  # Returns embedding dimension
    
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
        """Embed single amino acid"""
        tokens = self.tokenizers.sequence.encode(aa)
        token_tensor = torch.tensor(tokens).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(sequence_tokens=token_tensor)
        return outputs.embeddings[0, 1:-1].mean(dim=0)  # Remove BOS/EOS and average
    
    def embed_aa_pair(self, pair: str) -> torch.Tensor:
        """Embed AA pair and return 2*embed_dim vector"""
        tokens = self.tokenizers.sequence.encode(pair)
        token_tensor = torch.tensor(tokens).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(sequence_tokens=token_tensor)
        # Take mean of each AA's embeddings separately
        aa1_emb = outputs.embeddings[0, 1:2].mean(dim=0)  # First AA
        aa2_emb = outputs.embeddings[0, 2:3].mean(dim=0)  # Second AA
        return torch.cat([aa1_emb, aa2_emb])

def process_with_esm3(csv_path: str, output_dir: str, device: str = 'cpu'):
    embedder = ESM3Embedder(device)
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
        "output_dir": "data/esm3_processed_embeddings_2d",
        "device": "cuda:1" if torch.cuda.is_available() else "cpu"
    }
    process_with_esm3(**config)
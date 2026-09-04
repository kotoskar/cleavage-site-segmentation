# ACTIVATE YOUR ENVIRONMENT FROM THE COMMAND LINE FIRST!
# conda activate clustering_env
import os
from Bio import SeqIO
import subprocess

# File paths
input_ids_path = 'protein_id_homo.txt'         # File with the list of UniProt IDs
fasta_input_path = os.environ.get('DEEPPEPTIDE_FASTA', 'data/protein_sequences.fasta')  # Source FASTA file with sequences
fasta_output_path = 'sequences_homo.fasta'     # Where to save the matched sequences

# 1. Read UniProt IDs from file
with open(input_ids_path, 'r') as f:
    uniprot_ids = set(line.strip() for line in f if line.strip())

# 2. Load sequences from the fasta file and filter to the requested IDs
filtered_seqs = (
    rec for rec in SeqIO.parse(fasta_input_path, 'fasta')
    if rec.id in uniprot_ids
)

# 3. Save the matched sequences to a new fasta file
count = SeqIO.write(filtered_seqs, fasta_output_path, 'fasta')
print(f'Saved sequences: {count}')

# 4. Run graphpart needle via subprocess
command = [
    'graphpart', 'needle',
    '--fasta-file', fasta_output_path,
    '--threshold', '0.3',
    '--out-file', 'graphpart_assignments_homo.csv',
    '--labels-name', 'label',
    '--partitions', '5',
    '--threads', '4'
]

print("Running GraphPart command:")
try:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print("Command output:")
    print(result.stdout)
    if result.stderr:
        print("Command errors:", result.stderr)
except subprocess.CalledProcessError as e:
    print(f"Error running command: {e}")
    print(f"Error output: {e.stderr}")

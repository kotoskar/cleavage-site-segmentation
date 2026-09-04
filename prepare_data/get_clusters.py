import csv
from collections import defaultdict
from Bio import SeqIO

# File paths
csv_path = 'graphpart_assignments_homo.csv'
fasta_path = 'sequences_homo.fasta'

# 1. Read the AC -> cluster mapping from the csv
ac_to_cluster = {}
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        ac = row['AC']
        cluster = row['cluster']
        # Coerce cluster to int (values may come in as floats)
        cluster_id = int(float(cluster))
        ac_to_cluster[ac] = cluster_id

# 2. Load all sequences from the fasta file into a dict keyed by ID
seqs = SeqIO.to_dict(SeqIO.parse(fasta_path, 'fasta'))

# 3. Group sequences by cluster
clusters = defaultdict(list)
for ac, cluster_id in ac_to_cluster.items():
    if ac in seqs:
        clusters[cluster_id].append(seqs[ac])
    else:
        print(f'Warning: sequence with ID {ac} not found in {fasta_path}')

# 4. Save each cluster's sequences to a separate file
for cluster_id, records in clusters.items():
    out_filename = f'cluster_{cluster_id}.txt'
    count = SeqIO.write(records, out_filename, 'fasta')
    print(f'Cluster {cluster_id}: saved {count} sequences to file {out_filename}')

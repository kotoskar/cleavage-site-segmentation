import pandas as pd

df = pd.read_csv('data/uniprot_2022/labeled_sequences.csv')
partition = pd.read_csv('data/uniprot_2022/graphpart_assignments.csv')

homo_ids = df.loc[df['organism'] == "Homo sapiens"]['protein_id']
with open('data/uniprot_2022/protein_id_homo.txt', 'w') as f:
    f.write('\n'.join(homo_ids))

subset25 = set(partition.groupby('cluster')['AC'].sample(frac=0.25).agg(str))
subset50 = set(partition.groupby('cluster')['AC'].sample(frac=0.5).agg(str))
subset75 = set(partition.groupby('cluster')['AC'].sample(frac=0.75).agg(str))

df[df['protein_id'].isin(subset25)].drop('Unnamed: 0',axis=1).to_csv('data/uniprot_2022/labeled_sequences25.csv')
df[df['protein_id'].isin(subset50)].drop('Unnamed: 0',axis=1).to_csv('data/uniprot_2022/labeled_sequences50.csv')
df[df['protein_id'].isin(subset75)].drop('Unnamed: 0',axis=1).to_csv('data/uniprot_2022/labeled_sequences75.csv')
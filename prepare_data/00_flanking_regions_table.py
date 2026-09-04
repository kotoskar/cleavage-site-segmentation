# Aim of this script is to get flanking regions from peptides and propeptides of AA sequences
import csv
import re

def get_flanking_regions(sequence, coord_string, protein_id, protein_name, region_type):
    """Extract flanking regions (2 residues before start and after end)"""
    if not coord_string or str(coord_string).strip().lower() in ('', 'none', 'nan'):
        return []
    
    coord_pairs = re.findall(r'\((\d+)-(\d+)\)', coord_string)
    flanking_data = []
    
    for start, end in coord_pairs:
        start_idx = int(start) - 1  # Convert to 0-based
        end_idx = int(end)
        
        # Calculate flanking regions
        pre_start = max(0, start_idx - 2)
        post_end = min(len(sequence), end_idx + 2)
        
        # Extract the regions
        pre_region = sequence[pre_start:start_idx]  # 2 residues before start
        post_region = sequence[end_idx:post_end]    # 2 residues after end
        
        flanking_data.append({
            'protein_id': protein_id,
            'protein_name': protein_name,
            'coordinates': f"{start}-{end}",
            'pre-region': pre_region,
            'post-region': post_region,
            'type': region_type,
            'length': len(sequence)
        })
    
    return flanking_data

# Initialize output table
output_table = []

with open('data/tables_merged.csv', mode='r', newline='') as file:
    csv_reader = csv.DictReader(file)
    
    for row in csv_reader:
        protein_id = row['AC']
        protein_name = row['protein_name']
        sequence = row['sequence']
        
        # Process peptide coordinates
        if 'coordinates' in row:
            output_table.extend(get_flanking_regions(
                sequence, row['coordinates'], protein_id, protein_name, 'peptide'
            ))
        
        # Process propeptide coordinates
        if 'propeptide_coordinates' in row:
            output_table.extend(get_flanking_regions(
                sequence, row['propeptide_coordinates'], protein_id, protein_name, 'propeptide'
            ))

# Print the table header
print("protein_id,protein_name,coordinates,pre-region,post-region,type,length")

# Print each row of the table
for row in output_table:
    print(f"{row['protein_id']},{row['protein_name']},{row['coordinates']},{row['pre-region']},{row['post-region']},{row['type']},{row['length']}")

# Define the output file path
output_file_path = 'data/flanking_regions.csv'

# Write the output_table to a CSV file
with open(output_file_path, mode='w', newline='') as csvfile:
    # Define the fieldnames/columns
    fieldnames = ['protein_id', 'protein_name', 'coordinates', 
                 'pre-region', 'post-region', 'type', 'length']
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write the header
    writer.writeheader()
    
    # Write all rows
    writer.writerows(output_table)

print(f"Results successfully saved to: {output_file_path}")

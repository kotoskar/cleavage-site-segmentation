import csv

def fix_region(region: str, is_pre: bool) -> str:
    """
    Normalizes a pre-region or post-region field according to these rules:
    - If empty, returns '@@'
    - If one character, pads with '@' on the left (pre-region) or right (post-region)
    - If two characters, returns as-is
    """
    if not region or region.strip() == '':
        return '@@'
    region = region.strip()
    if len(region) == 1:
        if is_pre:
            return '@' + region
        else:
            return region + '@'
    if len(region) == 2:
        return region
    # If longer than 2, leave as-is (or truncate, depending on downstream use).
    return region

input_file = 'flanking_regions.csv'    # replace with your input file name
output_file = 'flanking_regions_no_missing.csv'

with open(input_file, newline='', encoding='utf-8') as csv_in, \
     open(output_file, 'w', newline='', encoding='utf-8') as csv_out:

    reader = csv.DictReader(csv_in)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        row['pre-region'] = fix_region(row.get('pre-region', ''), is_pre=True)
        row['post-region'] = fix_region(row.get('post-region', ''), is_pre=False)
        writer.writerow(row)

print(f"Done. Result saved to {output_file}")

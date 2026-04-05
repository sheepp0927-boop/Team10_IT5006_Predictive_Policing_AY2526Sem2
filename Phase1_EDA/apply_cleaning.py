import pandas as pd

# --- CONFIGURATION ---
RAW_DATA_PATH = 'Crime_Dataset.csv'
ANNOTATED_FILE = 'cleaning_candidates.csv' # The file you edited in Excel
OUTPUT_FILE = 'Crime_Dataset_Cleaned.csv'

print("--- 1. Loading Data & Annotations... ---")
df = pd.read_csv(RAW_DATA_PATH)
annotations = pd.read_csv(ANNOTATED_FILE)

# Filter for only the rows you marked as "MERGE" (case-insensitive)
# It will look for 'MERGE', 'merge', 'Merge', etc.
to_merge = annotations[annotations['Action'].str.upper() == 'MERGE']

print(f"Found {len(to_merge)} approved merges out of {len(annotations)} candidates.")

if len(to_merge) == 0:
    print("No merges found! Did you save your Excel file with 'MERGE' in the Action column?")
    exit()

print("\n--- 2. Applying Your Fixes... ---")

# Create a dictionary for mapping: { 'Column_Name': { 'Bad_Value': 'Good_Value' } }
merge_dict = {}

for index, row in to_merge.iterrows():
    col = row['Column']
    bad_val = row['Original_Value']
    good_val = row['Suggested_Fix']
    
    if col not in merge_dict:
        merge_dict[col] = {}
    
    merge_dict[col][bad_val] = good_val

# Apply the replacements column by column
for col, mapping in merge_dict.items():
    # Calculate impact before merging
    affected_rows = df[col].isin(mapping.keys()).sum()
    print(f"   Column '{col}': Merging {len(mapping)} values (affecting {affected_rows:,} rows)...")
    
    # Perform the merge
    df[col] = df[col].replace(mapping)

# --- 3. SAVING ---
print(f"\nSaving cleaned dataset to '{OUTPUT_FILE}'...")
df.to_csv(OUTPUT_FILE, index=False)
print("Done! Your manual annotations have been applied.")
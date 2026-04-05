import pandas as pd
import difflib

# --- CONFIGURATION ---
FILE_PATH = 'Crime_Dataset.csv'
OUTPUT_CSV = 'cleaning_candidates.csv'

# Columns to scan for typos/inconsistencies
cols_to_check = ['Location Description', 'Description', 'Primary Type']

print("Generating cleaning candidates for manual annotation...")
df = pd.read_csv(FILE_PATH, usecols=cols_to_check)

candidates = []

for col in cols_to_check:
    # Get all unique values sorted by frequency (descending)
    val_counts = df[col].value_counts()
    all_values = val_counts.index.tolist()
    
    # Iterate through every value to find "better" (more frequent) matches
    for i, current_val in enumerate(all_values):
        # Only compare against values that are MORE frequent (higher in the list)
        more_frequent_values = all_values[:i]
        if not more_frequent_values: continue
        
        current_str = str(current_val)
        
        # Fuzzy match: Look for 90% similarity
        matches = difflib.get_close_matches(current_str, [str(v) for v in more_frequent_values], n=1, cutoff=0.90)
        
        if matches:
            better_match = matches[0]
            candidates.append({
                'Column': col,
                'Original_Value': current_str,     # The less frequent value (potential typo)
                'Original_Count': val_counts[current_val],
                'Suggested_Fix': better_match,     # The more frequent value (potential correct spelling)
                'Fix_Count': val_counts[better_match],
                'Action': 'KEEP' # Default action (Change this to MERGE in Excel)
            })

# Save to CSV for you to open
pd.DataFrame(candidates).to_csv(OUTPUT_CSV, index=False)
print(f"Done! Open '{OUTPUT_CSV}' in Excel.")
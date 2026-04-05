import pandas as pd
import difflib

# --- CONFIGURATION ---
FILE_PATH = 'Crime_Dataset.csv'

# We focus ONLY on the text columns with high rare label counts (from your Table 4)
TARGET_COLUMNS = ['Location Description', 'Description', 'Primary Type']

print(f"--- Loading Data for Spell Check ---")
df = pd.read_csv(FILE_PATH, usecols=TARGET_COLUMNS)

print(f"Scanning columns: {', '.join(TARGET_COLUMNS)}...\n")

for col in TARGET_COLUMNS:
    print(f"🔹 ANALYZING: {col}")
    
    # 1. Get value counts
    val_counts = df[col].value_counts()
    
    # 2. Define "Common" (Correct) vs "Rare" (Suspicious)
    # Common: Appears at least 100 times
    # Rare: Appears less than 10 times
    common_values = val_counts[val_counts >= 100].index.tolist()
    rare_values = val_counts[val_counts < 10].index.tolist()
    
    # Skip if nothing to check
    if not rare_values:
        print(f"   (No rare labels found in {col}. Clean.)\n")
        continue

    # 3. Fuzzy Match
    potential_typos = []
    
    for rare in rare_values:
        # We handle strings only
        rare_str = str(rare)
        
        # Get closest match from 'common_values'
        # cutoff=0.85 means words must be 85% similar (allows 1-2 char difference)
        matches = difflib.get_close_matches(rare_str, [str(c) for c in common_values], n=1, cutoff=0.85)
        
        if matches:
            likely_correct = matches[0]
            potential_typos.append({
                'Rare (Typo?)': rare_str,
                'Count': val_counts[rare],
                'Suggested Correction': likely_correct
            })
    
    # 4. Report Results
    if potential_typos:
        print(f"   ⚠️ Found {len(potential_typos)} potential errors:")
        results_df = pd.DataFrame(potential_typos)
        # Display nicely in terminal
        print(results_df.to_string(index=False))
    else:
        print("   ✅ No obvious spelling errors found (Rare labels seem distinct).")
    
    print("-" * 50)

print("\nDone.")
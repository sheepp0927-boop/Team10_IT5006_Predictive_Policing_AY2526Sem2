import pandas as pd

# Load your CLEANED dataset (the one we just saved)
df = pd.read_csv('Crime_Dataset_Cleaned.csv')

# Get value counts
counts = df['Primary Type'].value_counts()

print("--- Primary Types & Counts ---")
print(counts)

# Optional: Save to CSV so you can open in Excel and move rows around to plan your groups
counts.to_csv('primary_type_counts.csv')
print("\nSaved list to 'primary_type_counts.csv' for planning.")
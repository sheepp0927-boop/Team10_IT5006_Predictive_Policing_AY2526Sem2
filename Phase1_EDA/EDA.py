import pandas as pd
import webbrowser
import os
import difflib

# --- CONFIGURATION ---
FILE_PATH = 'Crime_Dataset_Cleaned.csv' 
OUTPUT_FILE = 'EDA_Comprehensive_Report_Cleaned.html'

print("--- 1. Loading Data... ---")
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"ERROR: Could not find '{FILE_PATH}'. Check spelling!")
    exit()

# Parse dates efficiently
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

# --- 2. PREPARING DATA TABLES ---

# [TABLE 1] Executive Summary
summary_data = {
    'Metric': [
        'Total Rows', 'Total Columns', 'Duplicate Rows', 
        'Start Date', 'End Date', 'Time Span', 
        'Arrest Rate', 'Domestic Incident Rate'
    ],
    'Value': [
        f"{df.shape[0]:,}", 
        df.shape[1], 
        f"{df.duplicated().sum():,}",
        df['Date'].min().strftime('%Y-%m-%d'),
        df['Date'].max().strftime('%Y-%m-%d'),
        f"{(df['Date'].max() - df['Date'].min()).days:,} days",
        f"{(df['Arrest'].mean() * 100):.2f}%",
        f"{(df['Domestic'].mean() * 100):.2f}%"
    ]
}
summary_df = pd.DataFrame(summary_data)

# [TABLE 2] Temporal Integrity
year_counts = df['Date'].dt.year.value_counts().sort_index()
year_df = pd.DataFrame({
    'Year': year_counts.index,
    'Row Count': year_counts.values,
    '% of Total': (year_counts.values / len(df) * 100).round(2)
})

# [TABLE 3] Missing Data Profile
regex_pattern = r'^(unknown|n/a|none|\?|-|0|\*)$'
missing_data = []

for col in df.columns:
    nan_count = df[col].isnull().sum()
    hidden_count = 0
    if df[col].dtype == 'object':
        matches = df[col].astype(str).str.strip().str.contains(
            regex_pattern, regex=True, case=False, na=False
        )
        hidden_count = matches.sum()

    total_defective = nan_count + hidden_count
    if total_defective > 0:
        missing_data.append({
            'Column Name': col,
            'Empty Cells (Blanks)': nan_count,
            'Hidden (Regex Matches)': hidden_count,
            '% Defective': (total_defective / len(df) * 100).round(4)
        })

if missing_data:
    missing_df = pd.DataFrame(missing_data).sort_values('% Defective', ascending=False)
else:
    missing_df = pd.DataFrame({'Message': ['No Missing or Hidden Data Found']})

# [TABLE 4] Variable Cardinality
target_cols = [
    'Primary Type', 'Location Description', 'District', 'Beat', 'FBI Code',
    'IUCR', 'Description', 'Arrest', 'Domestic', 'Ward', 'Community Area'
]

cardinality_data = []
for col in target_cols:
    if col in df.columns:
        val_counts = df[col].value_counts()
        rare_count = (val_counts < 10).sum()
        cardinality_data.append({
            'Column Name': col,
            'Unique Values': df[col].nunique(),
            'Data Type': str(df[col].dtype),
            'Top Value (Mode)': str(df[col].mode()[0]) if not df[col].mode().empty else "N/A",
            'Rare Labels (<10)': rare_count
        })
card_df = pd.DataFrame(cardinality_data)

# [TABLE 5] Data Hygiene (Full Spectrum Typo Scan) - IMPROVED
typo_cols = ['Location Description', 'Description', 'Primary Type']
typo_findings = []

print("--- Scanning ALL values for Typos (Full Spectrum Check)... ---")

for col in typo_cols:
    if col in df.columns:
        # Get ALL unique values and their counts, sorted by frequency (most common first)
        val_counts = df[col].value_counts()
        all_values = val_counts.index.tolist()
        
        # We iterate through every value (except the very top one)
        for i, current_val in enumerate(all_values):
            # Optimization: Only check against values that are MORE frequent than the current one
            # (Because we assume the more frequent one is the 'correct' version)
            # We look at the slice of values *before* the current one in our sorted list
            more_frequent_values = all_values[:i]
            
            # Skip if we are at the top (nothing more frequent)
            if not more_frequent_values:
                continue

            current_str = str(current_val)
            
            # Use fuzzy matching against the more frequent list
            # We increase cutoff to 0.90 to reduce false positives since we are checking everything
            matches = difflib.get_close_matches(current_str, [str(v) for v in more_frequent_values], n=1, cutoff=0.90)
            
            if matches:
                better_match = matches[0]
                
                # Logic Check: Only flag if the better match is significantly more frequent?
                # For now, we just list it.
                typo_findings.append({
                    'Column': col,
                    'Suspicious Value': current_str,
                    'Count': val_counts[current_val],
                    'Did you mean? (More Frequent)': better_match,
                    'Freq Match': val_counts[better_match] 
                })

if typo_findings:
    typo_df = pd.DataFrame(typo_findings)
    # Sort by 'Count' descending to show the biggest "Systematic Errors" first
    typo_df = typo_df.sort_values('Count', ascending=False)
else:
    typo_df = pd.DataFrame({'Message': ['No fuzzy-match typos detected.']})

# --- 3. STYLING THE TABLES (HTML/CSS) ---

def style_table(styler, title):
    styler.set_caption(title)
    styler.set_table_styles([
        {'selector': 'thead th', 'props': [('background-color', '#4a4a4a'), ('color', 'white'), ('font-family', 'Arial'), ('text-align', 'left')]},
        {'selector': 'tbody td', 'props': [('font-family', 'Arial'), ('border', '1px solid #ddd'), ('padding', '8px')]},
        {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-size', '16px'), ('font-weight', 'bold'), ('margin-bottom', '10px'), ('text-align', 'left')]}
    ])
    styler.hide(axis='index')
    return styler.to_html()

html_summary = style_table(summary_df.style, "1. Executive Summary")
html_year = style_table(year_df.style.background_gradient(cmap='Greens', subset=['Row Count']), "2. Temporal Integrity")
html_missing = style_table(missing_df.style.background_gradient(cmap='Reds', subset=['% Defective']), "3. Missing Data Profile")
html_card = style_table(card_df.style.background_gradient(cmap='Blues', subset=['Unique Values']), "4. Variable Cardinality")

# Styling for Typos - Highlight high-frequency errors
if not typo_df.empty and 'Count' in typo_df.columns:
    html_typo = style_table(typo_df.style.background_gradient(cmap='Oranges', subset=['Count']), "5. Data Hygiene Check (Full Spectrum)")
else:
    html_typo = style_table(typo_df.style, "5. Data Hygiene Check (Full Spectrum)")

# --- 4. EXPORTING FINAL REPORT ---

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>EDA Comprehensive Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 40px; background-color: #f9f9f9; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        .container {{ display: flex; flex-direction: column; gap: 40px; max-width: 1100px; margin: auto; }}
        table {{ border-collapse: collapse; width: 100%; box-shadow: 0 0 10px rgba(0,0,0,0.1); background-color: white; }}
        th, td {{ padding: 12px; text-align: left; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .note {{ font-size: 0.9em; color: #666; font-style: italic; margin-top: -10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Chicago Crime Dataset: Comprehensive Profile</h1>
        {html_summary}
        {html_year}
        {html_missing}
        <div class="note">Note: "Hidden" checks for placeholders like '?', 'Unknown', or '0' using regex.</div>
        {html_card}
        {html_typo}
        <div class="note">Note: This section compares EVERY unique value against more frequent values to catch systematic errors.</div>
    </div>
</body>
</html>
"""

with open(OUTPUT_FILE, 'w') as f:
    f.write(html_content)

print(f"Report generated successfully: {OUTPUT_FILE}")
print("Opening in your browser...")
webbrowser.open('file://' + os.path.realpath(OUTPUT_FILE))
import pandas as pd
from IPython.display import display, HTML

# --- CONFIGURATION ---
# TIP: In VS Code, right-click your CSV file in the file explorer 
# and select "Copy Path" if you get a 'File Not Found' error.
FILE_PATH = 'Crime_Dataset.csv' 

# --- 1. LOAD DATA ---
print("Loading data...")
df = pd.read_csv(FILE_PATH)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

# --- 2. GENERATE REPORT TABLES ---

# A. The "Executive Summary" Table
summary_data = {
    'Metric': [
        'Total Rows', 'Total Columns', 'Duplicate Rows', 
        'Start Date', 'End Date', 'Time Span (Days)', 
        'Arrest Rate (%)', 'Domestic Incident Rate (%)'
    ],
    'Value': [
        f"{df.shape[0]:,}", 
        df.shape[1], 
        f"{df.duplicated().sum():,}",
        df['Date'].min().strftime('%Y-%m-%d'),
        df['Date'].max().strftime('%Y-%m-%d'),
        (df['Date'].max() - df['Date'].min()).days,
        f"{(df['Arrest'].mean() * 100):.2f}%",
        f"{(df['Domestic'].mean() * 100):.2f}%"
    ]
}
summary_df = pd.DataFrame(summary_data)

# B. The "Missing Data" Table
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if not missing.empty:
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        '% Total': (missing / len(df) * 100).round(2)
    })
else:
    missing_df = pd.DataFrame()

# C. The "Variable Profile" Table
target_cols = [
    'Primary Type', 'Location Description', 'District', 'Beat', 'FBI Code',
    'IUCR', 'Description', 'Arrest', 'Domestic', 'Ward', 'Community Area'
]

cardinality_data = []
for col in target_cols:
    if col in df.columns:
        cardinality_data.append({
            'Column Name': col,
            'Unique Values': df[col].nunique(),
            'Data Type': str(df[col].dtype),
            'Top Value (Mode)': df[col].mode()[0] if not df[col].mode().empty else "N/A"
        })

card_df = pd.DataFrame(cardinality_data)


# --- 3. DISPLAY OUTPUT (Nicely Formatted) ---

print("EXECUTIVE SUMMARY")
# We use .style to make it look like a report
display(summary_df.style.hide(axis='index')
        .set_properties(**{'text-align': 'left', 'font-size': '12pt', 'background-color': '#f4f4f4'})
        .set_table_styles([{'selector': 'th', 'props': [('text-align', 'left'), ('background-color', '#404040'), ('color', 'white')]}]))

print("\nMISSING DATA PROFILE")
if not missing_df.empty:
    display(missing_df.style
            .background_gradient(cmap='Reds', subset=['% Total'])
            .format({'% Total': "{:.2f}%"}))
else:
    print("✅ No missing data found!")

print("\nVARIABLE CARDINALITY & TYPES")
display(card_df.style.hide(axis='index')
        .background_gradient(cmap='Blues', subset=['Unique Values'])
        .set_properties(**{'text-align': 'left'}))
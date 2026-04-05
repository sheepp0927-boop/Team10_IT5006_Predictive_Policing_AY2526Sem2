import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

# --- CONFIGURATION ---
DATA_FOLDER = '.' 
CRIME_FILE = 'Crime_Dataset_Final.csv' 
CENSUS_FILES = [f for f in os.listdir(DATA_FOLDER) if f.startswith('CCA_') and f.endswith('.geojson')]
CENSUS_FILES.sort()

# ====================================================
# PART 1: ROBUST DATA LOADING (With Name Mapping)
# ====================================================
print("--- 1. BUILDING MAPPING & LOADING DATA ---")

# 1. Create Name-to-ID Map (To fix older files)
id_map = {}
try:
    # Find a newer file (2021+) to serve as the Master Key
    master_file = [f for f in CENSUS_FILES if int(''.join(filter(str.isdigit, f))) > 2020][-1]
    with open(os.path.join(DATA_FOLDER, master_file), 'r') as f:
        data = json.load(f)
    for feature in data['features']:
        p = feature['properties']
        if 'GEOG' in p and 'GEOID' in p:
            id_map[p['GEOG']] = int(p['GEOID'])
    print(f"✅ Created ID Mapping from {master_file}")
except:
    print("⚠️ Could not create ID map. Older files might be skipped.")

# 2. Load Longitudinal Data
master_df = pd.DataFrame()
try:
    df_crime_all = pd.read_csv(os.path.join(DATA_FOLDER, CRIME_FILE))
    df_crime_all['Year'] = pd.to_datetime(df_crime_all['Date']).dt.year
except:
    print("⚠️ Using Crime Lite...")
    df_crime_all = pd.read_csv(os.path.join(DATA_FOLDER, 'Crime_Dataset_Lite.csv'))
    df_crime_all['Year'] = pd.to_datetime(df_crime_all['Date']).dt.year

for filename in CENSUS_FILES:
    try:
        year = int(''.join(filter(str.isdigit, filename)))
        
        with open(os.path.join(DATA_FOLDER, filename), 'r') as f:
            data = json.load(f)
        rows = [f['properties'] for f in data['features']]
        df_year = pd.DataFrame(rows)
        
        # FIX MISSING IDs
        if 'GEOID' not in df_year.columns and 'GEOG' in df_year.columns:
            df_year['GEOID'] = df_year['GEOG'].map(id_map)

        # Standardize Columns
        mapping = {
            'GEOID': 'Community Area', 'GEOG_KEY': 'Community Area', 
            'TOT_POP': 'Population', 'MEDINC': 'Median_Income', 'MED_INC': 'Median_Income',
            'UNEMP': 'Unemp_Count', 'IN_LBFRC': 'Labor_Force', 
            'BACH': 'Bachelor_Count', 'POP_25OV': 'Adult_Pop',    
            'WHITE': 'White_Count', 'BLACK': 'Black_Count', 'HISP': 'Hisp_Count'
        }
        df_year = df_year.rename(columns=mapping)
        
        # Check Columns
        req = ['Median_Income', 'Unemp_Count', 'Labor_Force', 'Black_Count', 'Bachelor_Count', 'Adult_Pop', 'Community Area']
        if not all(c in df_year.columns for c in req): continue

        # Calculate Rates
        for c in req + ['Population']: df_year[c] = pd.to_numeric(df_year[c], errors='coerce')
        
        df_year['Unemployment_Rate'] = (df_year['Unemp_Count'] / df_year['Labor_Force']) * 100
        df_year['Bachelor_Rate'] = (df_year['Bachelor_Count'] / df_year['Adult_Pop']) * 100 
        df_year['Black_Pct'] = (df_year['Black_Count'] / df_year['Population']) * 100
        df_year['Year'] = year
        
        # Merge Crime
        crime_year = df_crime_all[df_crime_all['Year'] == year]
        if crime_year.empty: continue
        
        crime_counts = crime_year['Community Area'].value_counts().reset_index()
        crime_counts.columns = ['Community Area', 'Crime_Count']
        
        merged = df_year.merge(crime_counts, on='Community Area', how='inner')
        merged['Crime_Rate'] = (merged['Crime_Count'] / merged['Population']) * 1000
        
        master_df = pd.concat([master_df, merged], ignore_index=True)
        print(f"✅ Loaded {year}")
        
    except Exception as e:
        print(f"❌ Error {filename}: {e}")

analysis_df = master_df.dropna(subset=['Crime_Rate', 'Median_Income', 'Unemployment_Rate', 'Bachelor_Rate', 'Black_Pct'])
print(f"📊 DATA READY: {len(analysis_df)} Rows")

# ====================================================
# STEP 1: SIMPLE REGRESSION (With Explicit Residuals)
# ====================================================
print("\n--- STEP 1: SIMPLE REGRESSION DIAGNOSTICS ---")
predictors = ['Median_Income', 'Unemployment_Rate', 'Bachelor_Rate', 'Black_Pct']

# 1A. Regression Line Plots
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, var in enumerate(predictors):
    sns.regplot(data=analysis_df, x=var, y='Crime_Rate', ax=axes[i], 
                scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})
    axes[i].set_title(f"Trend: {var}")
plt.tight_layout()
plt.savefig('Step1_Trends.png')

# 1B. EXPLICIT RESIDUAL PLOTS (The Missing Piece)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, var in enumerate(predictors):
    X = analysis_df[[var]]
    y = analysis_df['Crime_Rate']
    model = sm.OLS(y, sm.add_constant(X)).fit()
    
    sns.scatterplot(x=analysis_df[var], y=model.resid, ax=axes[i], color='orange', alpha=0.5)
    axes[i].axhline(0, color='black', linestyle='--')
    axes[i].set_title(f"Residuals: {var}")
    axes[i].set_ylabel("Error")

plt.tight_layout()
plt.savefig('Step1_Explicit_Residuals.png')
print("✅ Saved Step 1 Plots (Trends + Residuals)")

# ====================================================
# STEP 2: MULTIPLE REGRESSION (Proof of Failure)
# ====================================================
print("\n--- STEP 2: MULTIPLE REGRESSION FAILURE ---")
X_multi = analysis_df[predictors]
X_multi = sm.add_constant(X_multi)
y = analysis_df['Crime_Rate']

model_multi = sm.OLS(y, X_multi).fit()

# Print Key Metrics
print(f"Adjusted R-Squared: {model_multi.rsquared_adj:.3f}")
print("P-Values (Note Income):")
print(model_multi.pvalues.round(4))

# VIF Calculation
vif_data = pd.DataFrame()
vif_data["Variable"] = X_multi.columns
vif_data["VIF"] = [variance_inflation_factor(X_multi.values, i) for i in range(X_multi.shape[1])]
print("\nVIF Scores:\n", vif_data[vif_data['Variable'] != 'const'])

# Residual Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=model_multi.fittedvalues, y=model_multi.resid, color='purple', alpha=0.5)
plt.axhline(0, color='black', linestyle='--')
plt.title("Step 2: Multiple Regression Residuals")
plt.savefig('Step2_Residuals.png')
print("✅ Saved Step 2 Plots")

# ====================================================
# STEP 3: OPTIMIZED CLUSTERING (Log-Income)
# ====================================================
print("\n--- STEP 3: CLUSTERING OPTIMIZATION ---")

# Create Log Income
analysis_df['Log_Income'] = np.log(analysis_df['Median_Income'])

# Compare Models to Maximise Score
model_options = {
    "Base": ['Median_Income', 'Unemployment_Rate', 'Black_Pct'],
    "Log": ['Log_Income', 'Unemployment_Rate', 'Black_Pct'],
    "Log+Edu": ['Log_Income', 'Unemployment_Rate', 'Black_Pct', 'Bachelor_Rate']
}

best_score = -1
best_model_name = ""
best_labels = None

for name, feats in model_options.items():
    X = analysis_df[feats]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    
    print(f"Model {name}: Silhouette = {score:.3f}")
    
    if score > best_score:
        best_score = score
        best_model_name = name
        best_labels = labels

print(f"\n🏆 WINNER: {best_model_name} (Score: {best_score:.3f})")

# Final Profile & Plot
analysis_df['Cluster'] = best_labels
profile = analysis_df.groupby('Cluster')[['Median_Income', 'Unemployment_Rate', 'Bachelor_Rate', 'Black_Pct', 'Crime_Rate']].mean()
print("\n--- FINAL CLUSTER PROFILES ---")
print(profile.round(1))

plt.figure(figsize=(10, 6))
sns.scatterplot(data=analysis_df, x='Median_Income', y='Unemployment_Rate', 
                hue='Cluster', palette='viridis', style='Cluster', s=100, alpha=0.8)
plt.xscale('log')
plt.title(f"Step 3: The '3 Chicagos' ({best_model_name})")
plt.xlabel("Median Income (Log Scale)")
plt.savefig('Step3_Final_Clusters.png')
print("✅ Saved Step 3 Plots")
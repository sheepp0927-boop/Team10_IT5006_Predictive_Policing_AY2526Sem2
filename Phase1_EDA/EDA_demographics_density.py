import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
import statsmodels.stats.diagnostic as diag
import numpy as np
import os

# --- CONFIGURATION ---
DATA_FOLDER = '.' 
CRIME_FILE = 'Crime_Dataset_Final.csv'
GEOJSON_BOUNDARIES = 'Boundaries.geojson'
CENSUS_2024 = 'CCA_2024.geojson'
CATS = ['Violent', 'Property', 'Vice', 'Public Order', 'Sexual', 'Other']

# --- 1. DATA LOADING ---
print("--- 1. Loading 2024 Base Data ---")

if not os.path.exists(CENSUS_2024):
    print(f"Error: {CENSUS_2024} not found.")
    exit()

gdf_census = gpd.read_file(CENSUS_2024)
df_census = pd.DataFrame(gdf_census.drop(columns='geometry', errors='ignore'))

def get_col(df, candidates):
    for c in df.columns:
        if c.upper() in [cand.upper() for cand in candidates]: return c
    return None

name_col = get_col(df_census, ['GEOG', 'COMMUNITY', 'NAME'])
id_col = get_col(df_census, ['GEOID', 'GEOG_KEY'])
pop_col = get_col(df_census, ['TOT_POP', 'POPULATION'])
inc_col = get_col(df_census, ['MEDINC', 'MED_INC', 'INCOME'])
unemp_col = get_col(df_census, ['UNEMP', 'UNEMP_COUNT'])
lbfrc_col = get_col(df_census, ['IN_LBFRC', 'LABOR_FORCE'])
black_col = get_col(df_census, ['BLACK', 'BLACK_COUNT'])
white_col = get_col(df_census, ['WHITE', 'POP_WHITE'])
hisp_col = get_col(df_census, ['HISP', 'POP_HISP', 'HISPANIC'])
asian_col = get_col(df_census, ['ASIAN', 'POP_ASIAN'])

census_clean = df_census[[name_col, id_col, pop_col, inc_col, unemp_col, lbfrc_col, black_col, white_col, hisp_col, asian_col]].copy()
census_clean.columns = ['Community', 'ID', 'Population', 'Income', 'Unemp_Count', 'Labor_Force', 'Black_Count', 'White_Count', 'Hisp_Count', 'Asian_Count']

# Numeric conversions and rate calculations
census_clean['ID'] = pd.to_numeric(census_clean['ID'], errors='coerce').fillna(0).astype(int)
census_clean['Income'] = pd.to_numeric(census_clean['Income'].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
census_clean['Unemployment_Rate'] = (pd.to_numeric(census_clean['Unemp_Count']) / pd.to_numeric(census_clean['Labor_Force']).replace(0, 1)) * 100
census_clean['Black_Pct'] = (pd.to_numeric(census_clean['Black_Count']) / pd.to_numeric(census_clean['Population']).replace(0, 1)) * 100
census_clean['White_Pct'] = (pd.to_numeric(census_clean['White_Count']) / pd.to_numeric(census_clean['Population']).replace(0, 1)) * 100
census_clean['Hisp_Pct'] = (pd.to_numeric(census_clean['Hisp_Count']) / pd.to_numeric(census_clean['Population']).replace(0, 1)) * 100
census_clean['Asian_Pct'] = (pd.to_numeric(census_clean['Asian_Count']) / pd.to_numeric(census_clean['Population']).replace(0, 1)) * 100

try:
    df_crime = pd.read_csv(CRIME_FILE)
    df_crime['Year'] = pd.to_datetime(df_crime['Date']).dt.year
    crime_2024 = df_crime[df_crime['Year'] == 2024]
except FileNotFoundError:
    print(f"Error: {CRIME_FILE} not found.")
    exit()

ref_gdf = gpd.read_file(GEOJSON_BOUNDARIES)
ref_id_col = 'area_num_1' if 'area_num_1' in ref_gdf.columns else 'area_numbe'
ref_gdf[ref_id_col] = ref_gdf[ref_id_col].astype(int)

# --- 2. ANALYSIS LOOP ---
for cat in ['All'] + CATS:
    print(f"Processing Category: {cat}...")
    
    crime_cat = crime_2024 if cat == 'All' else crime_2024[crime_2024['Crime_Category'] == cat]
    if crime_cat.empty: continue
    
    counts = crime_cat.groupby('Community Area').size().reset_index(name='Crimes')
    full_df = census_clean.merge(counts, left_on='ID', right_on='Community Area', how='left').fillna(0)
    full_df['Crime_Rate'] = (full_df['Crimes'] / full_df['Population']) * 1000
    top_20 = full_df.nlargest(20, 'Crime_Rate')['ID'].tolist()

    features_to_plot = [
        ('White_Pct', 'blue'), ('Black_Pct', 'purple'), 
        ('Hisp_Pct', 'orange'), ('Asian_Pct', 'green'),
        ('Income', 'teal'), ('Unemployment_Rate', 'brown')
    ]

    # --- FIGURE 1: REGRESSION ANALYSIS (3x2) ---
    fig1, axes1 = plt.subplots(3, 2, figsize=(16, 18))
    for i, (feat, color) in enumerate(features_to_plot):
        ax = axes1.flat[i]
        X = sm.add_constant(full_df[feat])
        res = sm.OLS(full_df['Crime_Rate'], X).fit()
        _, bp_p, _, _ = diag.het_breuschpagan(res.resid, res.model.exog)
        sns.regplot(data=full_df, x=feat, y='Crime_Rate', ax=ax, color=color, scatter_kws={'alpha':0.5})
        ax.set_ylim(bottom=0)
        status = "HETEROSCEDASTIC" if bp_p < 0.05 else "HOMOSCEDASTIC"
        ax.text(0.05, 0.82, f"R2: {res.rsquared:.2f}\nBP p: {bp_p:.3f}\nStatus: {status}", 
                transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        ax.set_title(f'{feat.replace("_Pct"," %").replace("_Rate"," Rate")} vs. {cat} Rate', fontweight='bold')
    plt.suptitle(f'Figure 1: {cat} Crime - Univariate Correlations (2024)', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'Evolution_{cat}_1_Stats.png')
    plt.close()

    # --- FIGURE 2: SPATIAL OVERLAP (3x2) ---
    map_gdf = ref_gdf.merge(full_df, left_on=ref_id_col, right_on='ID', how='left')
    fig2, axes2 = plt.subplots(3, 2, figsize=(16, 18))
    spatial_features = [
        ('White_Pct', 'Blues'), ('Black_Pct', 'Purples'), 
        ('Hisp_Pct', 'Oranges'), ('Asian_Pct', 'Greens'),
        ('Income', 'YlGn'), ('Unemployment_Rate', 'YlOrRd')
    ]
    for idx, (feat, cmap) in enumerate(spatial_features):
        ax = axes2.flat[idx]
        map_gdf.plot(column=feat, cmap=cmap, legend=True, ax=ax)
        map_gdf[map_gdf['ID'].isin(top_20)].boundary.plot(ax=ax, color='red', linewidth=1.8)
        ax.set_title(f'Spatial Map: {feat.replace("_Pct"," Pop %").replace("_Rate"," Rate")}', fontweight='bold')
        ax.axis('off')
    plt.suptitle(f'Figure 2: {cat} Crime Hotspots (Red) vs. Socioeconomic Factors', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'Evolution_{cat}_2_Overlap.png')
    plt.close()

    # --- FIGURE 3: CLUSTERING ---
    cluster_feats = ['Income', 'Unemployment_Rate', 'Black_Pct']
    cluster_data = full_df[['ID', 'Community'] + cluster_feats].dropna()
    X_scaled = StandardScaler().fit_transform(cluster_data[cluster_feats])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_data['Cluster'] = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, cluster_data['Cluster'])
    ranks = cluster_data.groupby('Cluster')['Income'].mean().sort_values(ascending=False).index
    rank_map = {old: new for new, old in enumerate(ranks)}
    cluster_data['Cluster'] = cluster_data['Cluster'].map(rank_map)
    profile = cluster_data.groupby('Cluster')[cluster_feats].mean()
    cluster_colors = {0: '#66c2a5', 1: '#fc8d62', 2: '#8da0cb'}
    labels = {i: f"Cluster {i}: ${profile.loc[i, 'Income']/1000:.0f}k Inc, {profile.loc[i, 'Unemployment_Rate']:.1f}% Unp, {profile.loc[i, 'Black_Pct']:.1f}% Blk" for i in range(3)}
    map_gdf_c = ref_gdf.merge(cluster_data, left_on=ref_id_col, right_on='ID', how='left')
    map_gdf_c['Color'] = map_gdf_c['Cluster'].map(cluster_colors).fillna('#f0f0f0')
    fig3, ax = plt.subplots(figsize=(14, 10))
    map_gdf_c.plot(color=map_gdf_c['Color'], edgecolor='white', linewidth=0.5, ax=ax)
    map_gdf_c[map_gdf_c['ID'].isin(top_20)].boundary.plot(ax=ax, color='#333333', linewidth=1.5, linestyle='--')
    legend_patches = [mpatches.Patch(color=cluster_colors[i], label=labels[i]) for i in range(3)]
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1), title=f"Silhouette Score: {sil_score:.3f}")
    ax.set_title(f'Figure 3: Socioeconomic Archetypes for {cat} Crime', fontsize=18, fontweight='bold')
    ax.axis('off')
    plt.savefig(f'Evolution_{cat}_3_Cluster.png')
    plt.close()

    # --- FIGURE 4: MATCHING UNIVARIATE RESIDUALS (3x2) ---
    # Now Figure 4 perfectly mirrors Figure 1 to prove why each factor fails
    fig4, axes4 = plt.subplots(3, 2, figsize=(16, 18))
    for i, (feat, color) in enumerate(features_to_plot):
        ax = axes4.flat[i]
        X = sm.add_constant(full_df[feat])
        res = sm.OLS(full_df['Crime_Rate'], X).fit()
        sns.scatterplot(x=res.fittedvalues, y=res.resid, ax=ax, color='red', alpha=0.5)
        ax.axhline(0, color='black', linestyle='--')
        ax.set_title(f'Residuals: {feat.replace("_Pct"," %").replace("_Rate"," Rate")}', fontweight='bold')
        ax.set_xlabel('Predicted Value'); ax.set_ylabel('Residual')
    plt.suptitle(f'Figure 4: {cat} Crime - Univariate Residual Diagnostics', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'Evolution_{cat}_4_Univariate_Residuals.png')
    plt.close()

    # --- FIGURE 5: FINAL MULTIVARIATE MODEL DIAGNOSTIC ---
    X_final = sm.add_constant(full_df[cluster_feats].dropna())
    model_final = sm.OLS(full_df.loc[X_final.index, 'Crime_Rate'], X_final).fit()
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=model_final.fittedvalues, y=model_final.resid, color='darkred', alpha=0.6, ax=ax5)
    ax5.axhline(0, color='black', linestyle='--')
    ax5.set_title(f'Figure 5: Residual Plot for {cat} Final Multivariate Model', fontweight='bold')
    ax5.set_xlabel('Predicted Rate'); ax5.set_ylabel('Residual (Error)')
    proof_text = f"Full Model R2: {model_final.rsquared:.3f}\nResidual mean: {np.mean(model_final.resid):.2e}"
    ax5.text(0.05, 0.90, proof_text, transform=ax5.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.savefig(f'Evolution_{cat}_5_Final_Diagnostics.png')
    plt.close()

print("Analysis Complete. All matching residuals and 10-year logic diagnostics processed.")
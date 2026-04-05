import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_FILE = 'Crime_Dataset_Final.csv'
GEOJSON_FILE = 'Boundaries.geojson'
CATS = ['Violent', 'Property', 'Vice', 'Public Order', 'Sexual', 'Other']

print("--- Loading & Preparing Data ---")
try:
    df = pd.read_csv(DATA_FILE)
    gdf = gpd.read_file(GEOJSON_FILE)
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# 1. Clean Data
df = df.dropna(subset=['Community Area'])
df['Community Area'] = df['Community Area'].astype(int)

# 2. Handle GeoJSON column variations
if 'area_num_1' in gdf.columns:
    gdf['area_num_1'] = gdf['area_num_1'].astype(int)
    merge_col = 'area_num_1'
else:
    gdf['area_numbe'] = gdf['area_numbe'].astype(int)
    merge_col = 'area_numbe'

# ==========================================
# PART 1: OVERALL MAP & TABLE (Renamed)
# ==========================================
print("Generating Overall Map (Figure 1)...")

# Aggregate All Crimes
overall_counts = df.groupby('Community Area').size().reset_index(name='Crime_Count')
overall_gdf = gdf.merge(overall_counts, left_on=merge_col, right_on='Community Area', how='left').fillna(0)

# Plot Overall Map
fig1, ax1 = plt.subplots(figsize=(10, 10))
overall_gdf.plot(column='Crime_Count', cmap='Reds', linewidth=0.8, edgecolor='0.6', legend=True,
                 legend_kwds={'label': "Total Incidents (2014-2024)", 'orientation': "horizontal", 'shrink': 0.6, 'pad': 0.02}, ax=ax1)

# FIXED TITLE
ax1.set_title('Spatial Distribution: All Crimes by Community Area', fontsize=16, fontweight='bold', pad=20)
ax1.set_axis_off()
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Generate Overall Table (Figure 2)
print("Generating Overall Table (Figure 2)...")
top_20 = overall_gdf.nlargest(20, 'Crime_Count')[['community', 'Crime_Count']]
top_20['Rank'] = range(1, 21)
top_20['Crime_Count'] = top_20['Crime_Count'].apply(lambda x: f"{int(x):,}")
top_20['community'] = top_20['community'].str.title()
table_data = top_20[['Rank', 'community', 'Crime_Count']].values.tolist()

fig2, ax2 = plt.subplots(figsize=(6, 10))
ax2.axis('off')
the_table = ax2.table(cellText=table_data, colLabels=['Rank', 'Community Area', 'Incidents'],
                      colWidths=[0.15, 0.55, 0.3], loc='center', cellLoc='center', colColours=['#eeeeee']*3)
the_table.auto_set_font_size(False)
the_table.set_fontsize(11)
the_table.scale(1.0, 1.8)
for (row, col), cell in the_table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#d6d6d6')
        cell.set_height(0.06)
    elif row % 2 == 0:
        cell.set_facecolor('#f9f9f9')

# FIXED TITLE
ax2.set_title('Top 20 Community Areas by Crime Volume', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ==========================================
# PART 2: CATEGORY-SPECIFIC MAPS (Renamed)
# ==========================================
print("\n--- Generating Category Maps (Figures 3-8) ---")

# Specific colormaps for distinction
cat_cmaps = {
    'Violent': 'Reds',       
    'Property': 'Blues',     
    'Vice': 'Purples',       
    'Public Order': 'Oranges',
    'Sexual': 'PuRd',
    'Other': 'Greys'
}

for i, cat in enumerate(CATS):
    print(f"Generating Map for: {cat}...")
    
    # 1. Filter Data
    cat_df = df[df['Crime_Category'] == cat]
    
    # 2. Aggregate
    cat_counts = cat_df.groupby('Community Area').size().reset_index(name='Crime_Count')
    
    # 3. Merge
    cat_gdf = gdf.merge(cat_counts, left_on=merge_col, right_on='Community Area', how='left')
    cat_gdf['Crime_Count'] = cat_gdf['Crime_Count'].fillna(0)
    
    # 4. Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = cat_cmaps.get(cat, 'Reds')
    
    cat_gdf.plot(column='Crime_Count',
                 cmap=cmap,      
                 linewidth=0.8,
                 edgecolor='0.6',  
                 legend=True,
                 legend_kwds={'label': f"Total {cat} Incidents", 'orientation': "horizontal", 'shrink': 0.6, 'pad': 0.02},
                 ax=ax)
    
    # FIXED TITLE (Specific to Category AND Unit)
    ax.set_title(f'Spatial Distribution: {cat} Crimes by Community Area', fontsize=16, fontweight='bold', pad=20)
    ax.set_axis_off()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

print("✅ All 8 Figures Generated Successfully.")
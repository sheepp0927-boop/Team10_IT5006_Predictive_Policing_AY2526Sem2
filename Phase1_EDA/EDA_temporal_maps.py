import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- CONFIGURATION ---
DATA_FILE = 'Crime_Dataset_Final.csv'
GEOJSON_FILE = 'Boundaries.geojson'

print("--- 1. Loading Data ---")
try:
    df = pd.read_csv(DATA_FILE)
    gdf = gpd.read_file(GEOJSON_FILE)
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# Clean & Prepare
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df = df.dropna(subset=['Community Area'])
df['Community Area'] = df['Community Area'].astype(int)

if 'area_num_1' in gdf.columns:
    gdf['area_num_1'] = gdf['area_num_1'].astype(int)
    merge_col = 'area_num_1'
else:
    gdf['area_numbe'] = gdf['area_numbe'].astype(int)
    merge_col = 'area_numbe'

# Aggregate Data (Area x Year)
annual_counts = df.groupby(['Community Area', 'Year']).size().unstack(fill_value=0)

# ==========================================
# VISUALIZATION 1: SMALL MULTIPLES (Fixed Layout)
# ==========================================
print("Generating Small Multiples...")

years_to_plot = [2014, 2016, 2018, 2020, 2022, 2024]
cols = 3
rows = 2

fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
axes = axes.flatten()

vmin, vmax = 0, annual_counts.max().max()

for idx, year in enumerate(years_to_plot):
    if year in annual_counts.columns:
        ax = axes[idx]
        
        year_data = annual_counts[year].reset_index(name='Count')
        map_gdf = gdf.merge(year_data, left_on=merge_col, right_on='Community Area', how='left')
        map_gdf['Count'] = map_gdf['Count'].fillna(0)
        
        map_gdf.plot(column='Count', 
                     cmap='YlOrRd', 
                     linewidth=0.5,
                     edgecolor='0.5',
                     vmin=vmin, vmax=vmax, 
                     ax=ax)
        
        ax.set_title(f"Year {year}", fontsize=14, fontweight='bold')
        ax.axis('off')

# Colorbar
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, label='Annual Incident Count')

# --- THE FIX ---
plt.suptitle('Temporal Shift: Crime Intensity Over a Decade', fontsize=22, fontweight='bold')
# This reserves the top 5% of the page for the title
plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.show()

# ==========================================
# VISUALIZATION 2: PEAK YEAR MAP (Fixed Layout)
# ==========================================
print("Generating Peak Year Map...")

peak_years = annual_counts.idxmax(axis=1).reset_index(name='Peak_Year')
peak_gdf = gdf.merge(peak_years, left_on=merge_col, right_on='Community Area', how='left')

fig, ax = plt.subplots(figsize=(12, 12))

cmap = plt.cm.get_cmap('coolwarm', 11) 

peak_gdf.plot(column='Peak_Year', 
              cmap=cmap, 
              linewidth=0.8,
              edgecolor='0.4',
              legend=True,
              categorical=False, 
              legend_kwds={'label': "Year of Maximum Crime Volume", 
                           'orientation': "horizontal", 
                           'shrink': 0.6, 
                           'pad': 0.02},
              ax=ax)

ax.set_title('The Crisis Timeline: When Did Crime Peak?', fontsize=20, fontweight='bold', pad=20)
ax.axis('off')

# --- THE FIX ---
# Same logic, leaving space at the top
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print("✅ Visualizations Generated.")
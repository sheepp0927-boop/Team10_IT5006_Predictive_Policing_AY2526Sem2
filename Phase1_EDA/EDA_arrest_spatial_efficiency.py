import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import os

# --- CONFIGURATION ---
DATA_FILE = 'Crime_Dataset_Final.csv'
GEOJSON_FILE = 'Boundaries.geojson'

print("--- 1. Loading & Preparing Data ---")

# Load Crime Data
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"❌ Error: {DATA_FILE} not found.")
    exit()

# Filter for the most recent complete year (e.g., 2024) to match previous analysis
df['Year'] = pd.to_datetime(df['Date']).dt.year
target_year = df['Year'].max() # Automatically pick latest year
print(f"Analyzing Arrest Efficiency for Year: {target_year}")

df_year = df[df['Year'] == target_year]

# --- 2. Calculate Arrest Efficiency ---
# Group by Community Area
# Count total incidents AND total arrests (where Arrest == True)
crime_stats = df_year.groupby('Community Area').agg(
    Total_Crimes=('ID', 'count'),
    Total_Arrests=('Arrest', 'sum')
).reset_index()

# Calculate Arrest Rate (%)
crime_stats['Arrest_Rate'] = (crime_stats['Total_Arrests'] / crime_stats['Total_Crimes']) * 100

# Identify Top 20 High Crime Areas (for overlay)
top_crime_areas = crime_stats.nlargest(20, 'Total_Crimes')['Community Area'].tolist()

# --- 3. Merge with Geography ---
ref_gdf = gpd.read_file(GEOJSON_FILE)

# Fix IDs
id_col = 'area_num_1' if 'area_num_1' in ref_gdf.columns else 'area_numbe'
ref_gdf[id_col] = ref_gdf[id_col].astype(int)

# Merge
map_gdf = ref_gdf.merge(crime_stats, left_on=id_col, right_on='Community Area', how='left')
map_gdf['Arrest_Rate'] = map_gdf['Arrest_Rate'].fillna(0)

# Create Boundary for High Crime Areas
high_crime_gdf = map_gdf[map_gdf['Community Area'].isin(top_crime_areas)]

# ==========================================
# VISUALIZATION 1: THE JUSTICE GAP MAP
# ==========================================
print("Generating Arrest Efficiency Map...")

fig, ax = plt.subplots(figsize=(12, 10))

# Plot Arrest Rate
# Color Scheme: Red = Low Efficiency (Bad), Green = High Efficiency (Good)
map_gdf.plot(column='Arrest_Rate', 
             cmap='RdYlGn', 
             legend=True, 
             legend_kwds={'label': "Arrest Rate (%)", 'shrink': 0.6},
             edgecolor='white', 
             linewidth=0.5,
             ax=ax)

# Overlay: Top 20 High Crime Areas (Black Dashed Line)
high_crime_gdf.boundary.plot(ax=ax, color='black', linewidth=2, linestyle='--', label='High Crime Zones')

# Custom Legend
black_line = mlines.Line2D([], [], color='black', linewidth=2, linestyle='--', label='Top 20 High Crime Areas')
plt.legend(handles=[black_line], loc='lower left', fontsize=11)

ax.set_title(f'The Justice Gap: Arrest Efficiency by Neighborhood ({target_year})', fontsize=18, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('Arrest_Efficiency_Map.png', dpi=300)
plt.show()

# ==========================================
# VISUALIZATION 2: SCATTER (VOLUME VS. EFFICIENCY)
# ==========================================
print("Generating Correlation Scatter...")

plt.figure(figsize=(10, 6))

# Plot
sns.regplot(data=map_gdf, x='Total_Crimes', y='Arrest_Rate', color='darkred', scatter_kws={'alpha':0.6})

plt.title(f'System Strain: Does More Crime Mean Fewer Arrests?', fontsize=16, fontweight='bold')
plt.xlabel('Total Crime Volume', fontsize=12)
plt.ylabel('Arrest Efficiency (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Annotate outliers (Highest/Lowest efficiency)
top_eff = map_gdf.nlargest(3, 'Arrest_Rate')
low_eff = map_gdf.nsmallest(3, 'Arrest_Rate')

for idx, row in pd.concat([top_eff, low_eff]).iterrows():
    plt.text(row['Total_Crimes'], row['Arrest_Rate'], row['community'].title(), 
             fontsize=9, fontweight='bold', ha='right')

plt.tight_layout()
plt.savefig('Arrest_Efficiency_Scatter.png', dpi=300)
plt.show()

# Print Stats
corr = map_gdf['Total_Crimes'].corr(map_gdf['Arrest_Rate'])
print(f"\n--- Statistical Finding ---")
print(f"Correlation between Crime Volume and Arrest Rate: {corr:.2f}")
if corr < -0.3:
    print("👉 Insight: Strong Negative Correlation. The system is overwhelmed; high-crime areas have significantly lower justice outcomes.")
else:
    print("👉 Insight: Efficiency is relatively consistent, regardless of volume.")

print("\n✅ Arrest Efficiency Analysis Complete.")
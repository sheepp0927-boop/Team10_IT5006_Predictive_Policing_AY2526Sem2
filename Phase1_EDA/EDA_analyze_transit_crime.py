import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os

# --- CONFIGURATION ---
CRIME_FILE = 'Crime_Dataset_Final.csv'
STATION_FILE = 'CTA_stations.csv'
GEOJSON_FILE = 'Boundaries.geojson'

# --- 1. LOAD DATA ---
print("--- 1. Loading Dataset ---")
try:
    df_crime = pd.read_csv(CRIME_FILE)
    df_stations = pd.read_csv(STATION_FILE)
    chicago = gpd.read_file(GEOJSON_FILE)
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit()

# --- 2. FILTER FOR 2014-2024 ---
print("--- 2. Filtering for 2014-2024 Era ---")
# Convert Date to datetime efficiently
df_crime['Date'] = pd.to_datetime(df_crime['Date'], errors='coerce')
df_crime['Year'] = df_crime['Date'].dt.year

# Strict Filter: 2014 to 2024
df_filtered = df_crime[(df_crime['Year'] >= 2014) & (df_crime['Year'] <= 2024)].copy()
df_filtered = df_filtered.dropna(subset=['Latitude', 'Longitude'])

print(f"   Original Rows: {len(df_crime):,}")
print(f"   Filtered Rows (2014-2024): {len(df_filtered):,}")

# --- 3. PREPARE STATIONS ---
# Clean Station Coordinates
coords = df_stations['Location'].astype(str).str.replace(r'[()]', '', regex=True).str.split(',', expand=True)
df_stations['Latitude'] = coords[0].str.strip().astype(float)
df_stations['Longitude'] = coords[1].str.strip().astype(float)

# --- 4. GENERATE "THERMAL" INTENSITY MAP ---
print(f"--- 3. Plotting {len(df_filtered):,} incidents (High Intensity Mode) ---")

# Setup Poster-Sized Canvas (20x30 inches) with Black Background
fig, ax = plt.subplots(figsize=(20, 30), facecolor='black') 
ax.set_facecolor('black')

# Base Layer: Dark Grey for context (subtle)
chicago.plot(ax=ax, color='#1a1a1a', edgecolor='#333333', linewidth=1.5, zorder=1)

# CRIME LAYER: "MOLTEN" EFFECT
# Settings tuned for 10-year dataset to maximize "Glow"
# Color: OrangeRed (#ff4500) looks hotter/brighter on black than standard Red
ax.scatter(df_filtered['Longitude'], df_filtered['Latitude'], 
           color='#ff4500', 
           s=1.5,          # Larger points to fill gaps
           alpha=0.10,     # Higher alpha (10%) to build intensity fast
           edgecolor='none',
           zorder=2)

# TRANSIT LAYER: "NEON" STATIONS
# Cyan (#00d4ff) contrasts perfectly with the Orange/Red fire
# Outer Glow (Large, Transparent)
ax.scatter(df_stations['Longitude'], df_stations['Latitude'], 
           color='#00d4ff', s=250, marker='o', alpha=0.4, zorder=3, label='_nolegend_')
# Inner Core (Small, Solid White)
ax.scatter(df_stations['Longitude'], df_stations['Latitude'], 
           color='white', s=60, marker='o', zorder=4, label='CTA Stations')

# --- 5. FORMATTING ---
# Crop to City Limits
ax.set_xlim([-87.95, -87.50])
ax.set_ylim([41.60, 42.05])
ax.set_axis_off()

# Title
plt.title('Chicago Crime Density (2014-2024)\nSpatial Intersection with CTA Hubs', 
          color='white', fontsize=35, fontweight='bold', pad=40)

# Legend (Customized for Dark Mode)
leg = ax.legend(loc='upper right', frameon=True, fontsize=20, facecolor='black', edgecolor='white')
for text in leg.get_texts():
    text.set_color("white")
leg.legend_handles[0].set_color('white') # Fix icon color
leg.legend_handles[0].set_sizes([150])

# Save High-Res
output_file = 'Crime_Thermal_Map_2014_2024.png'
print(f"Saving {output_file}...")
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
plt.close()

print(f"✅ Success! Thermal map saved as {output_file}")
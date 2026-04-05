import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import BallTree

# --- CONFIGURATION ---
DATA_FOLDER = '.'
CRIME_FILE = 'Crime_Dataset_Final.csv' 
STATION_FILE = 'CTA_stations.csv'
RADIUS_METERS = 400   # 400m Catchment (approx 5 min walk)
MIN_INCIDENTS = 500   # Filter: Ignore rare crimes (e.g. "Ritualism") to avoid skewed %

# --- 1. LOAD DATA ---
print("Loading Data...")
try:
    cta_df = pd.read_csv(f"{DATA_FOLDER}/{STATION_FILE}")
except:
    print("❌ Station file not found.")
    exit()

# Clean Coordinates
coords = cta_df['Location'].astype(str).str.replace(r'[()]', '', regex=True).str.split(',', expand=True)
cta_df['Latitude'] = coords[0].str.strip().astype(float)
cta_df['Longitude'] = coords[1].str.strip().astype(float)
stations = cta_df.drop_duplicates(subset=['Latitude', 'Longitude'])

# Load Crime
try:
    df = pd.read_csv(f"{DATA_FOLDER}/{CRIME_FILE}", usecols=['Date', 'Primary Type', 'Latitude', 'Longitude'])
except:
    df = pd.read_csv(f"{DATA_FOLDER}/Crime_Dataset_Lite.csv", usecols=['Date', 'Primary Type', 'Latitude', 'Longitude'])

df = df.dropna(subset=['Latitude', 'Longitude'])
print(f"Analyzing {len(df):,} incidents...")

# --- 2. CALCULATE PROXIMITY ---
print(f"Calculating proximity ({RADIUS_METERS}m)...")
R_EARTH = 6371000

# Build Tree
station_rads = np.radians(stations[['Latitude', 'Longitude']].values)
tree = BallTree(station_rads, metric='haversine')

# Query Tree
crime_rads = np.radians(df[['Latitude', 'Longitude']].values)
dist_rads, _ = tree.query(crime_rads, k=1)
df['Dist_Meters'] = dist_rads * R_EARTH
df['Near_Station'] = df['Dist_Meters'] <= RADIUS_METERS

# --- 3. CALCULATE PROPORTIONS (The "Transit Dependency" Metric) ---
print("Computing crime proportions...")

# Group by Type and count (Total vs Near Station)
stats = df.groupby('Primary Type')['Near_Station'].agg(['count', 'sum'])
stats.columns = ['Total_Citywide', 'Count_Near_Transit']

# Calculate Percentage: (Near / Total) * 100
stats['Percent_Near_Transit'] = (stats['Count_Near_Transit'] / stats['Total_Citywide']) * 100

# Filter out rare crimes (noise)
stats = stats[stats['Total_Citywide'] >= MIN_INCIDENTS]

# Sort by Percentage (Who is most linked to transit?)
stats = stats.sort_values('Percent_Near_Transit', ascending=False)

# Display Table
print("\n--- TRANSIT DEPENDENCY RANKING ---")
print(stats[['Total_Citywide', 'Count_Near_Transit', 'Percent_Near_Transit']].head(15))

# --- 4. VISUALIZATION ---
plt.figure(figsize=(12, 8))

# Plot Top 15
top_15 = stats.head(15)
sns.barplot(x=top_15['Percent_Near_Transit'], y=top_15.index, palette='viridis')

plt.title(f'Proportion of Top 15 Crimes Occurring within {RADIUS_METERS}m of "L" Stations', fontsize=16, fontweight='bold')
plt.xlabel('Percentage (%) of Total Incidents', fontsize=12)
plt.ylabel('Primary Crime Type', fontsize=12)

# Add Labels
for i, v in enumerate(top_15['Percent_Near_Transit']):
    plt.text(v + 0.5, i, f"{v:.1f}%", va='center', fontweight='bold')

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
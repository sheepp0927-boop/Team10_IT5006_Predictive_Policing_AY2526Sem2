import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster

# 1. LOAD DATA
# We load a sample to keep the map fast. 
# If you want the full picture, remove nrows, but be warned: the map will be large!
file_path = 'Crime_Dataset.csv'
try:
    print("Loading data...")
    df = pd.read_csv(file_path) 
except FileNotFoundError:
    print("Error: File not found.")
    exit()

# 2. DATA CLEANING FOR SPATIAL ANALYSIS
# Drop rows where location is missing
df_map = df.dropna(subset=['Latitude', 'Longitude', 'Primary Type'])

# Filter data to make the map specific and readable
# Let's look at a recent year in the sample (or the whole sample if no years found)
if 'Year' in df_map.columns:
    latest_year = df_map['Year'].max()
    df_map = df_map[df_map['Year'] == latest_year]
    print(f"Mapping data for Year: {latest_year}")

# 3. INITIALIZE MAP
# Center the map on Chicago [Lat, Long]
chicago_coords = [41.8781, -87.6298]
m = folium.Map(location=chicago_coords, zoom_start=11, tiles='CartoDB dark_matter')

# --- LAYER 1: HEATMAP (Density of all crimes) ---
# This shows "hotspots" where crime is most frequent
heat_data = [[row['Latitude'], row['Longitude']] for index, row in df_map.iterrows()]
HeatMap(heat_data, radius=15, blur=20, name="Crime Density").add_to(m)

# --- LAYER 2: MARKERS (Specific Severe Crimes) ---
# Let's pinpoint "HOMICIDE" or "WEAPONS VIOLATION" specifically
severe_crimes = df_map[df_map['Primary Type'].isin(['HOMICIDE', 'WEAPONS VIOLATION'])]
marker_cluster = MarkerCluster(name="Severe Crimes (Markers)").add_to(m)

for index, row in severe_crimes.iterrows():
    # Create a popup with details
    popup_text = f"Type: {row['Primary Type']}<br>Date: {row['Date']}"
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup_text,
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(marker_cluster)

# Add Layer Control to toggle between Heatmap and Markers
folium.LayerControl().add_to(m)

# 4. SAVE MAP
output_file = 'chicago_crime_map.html'
m.save(output_file)
print(f"\nMap saved as '{output_file}'. Open this file in your browser to interact!")
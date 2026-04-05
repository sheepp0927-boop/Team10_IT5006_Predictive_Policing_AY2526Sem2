from sklearn.neighbors import BallTree

crime_path = "/content/drive/MyDrive/crime/Crimes_2014_2024_with_category_updated.csv"
crime = pd.read_csv(crime_path)

# Load parsed CTA data
cta_path = "/content/drive/MyDrive/crime/CTA_data_parsed.csv"
cta = pd.read_csv(cta_path)

print("Crime data shape:", crime.shape)
print("CTA data shape:", cta.shape)

# Convert crime coordinates to numeric
crime["Latitude"] = pd.to_numeric(crime["Latitude"], errors="coerce")
crime["Longitude"] = pd.to_numeric(crime["Longitude"], errors="coerce")

# Keep only rows with valid coordinates
crime_valid = crime.dropna(subset=["Latitude", "Longitude"]).copy()
cta_valid = cta.dropna(subset=["Latitude", "Longitude"]).copy()

print("Crime records with valid coordinates:", crime_valid.shape[0])
print("CTA stations with valid coordinates:", cta_valid.shape[0])

# Extract coordinates
crime_coords = crime_valid[["Latitude", "Longitude"]].to_numpy(dtype=float)
cta_coords = cta_valid[["Latitude", "Longitude"]].to_numpy(dtype=float)

# Convert to radians for haversine distance
crime_rad = np.radians(crime_coords)
cta_rad = np.radians(cta_coords)

# Build BallTree
tree = BallTree(cta_rad, metric="haversine")

# Calculate distance to nearest station
distances, _ = tree.query(crime_rad, k=1)

earth_radius = 6371000
crime_valid["distance_to_nearest_station"] = distances.flatten() * earth_radius

# Count stations within 500 meters
radius_meters = 500
radius_radians = radius_meters / earth_radius

crime_valid["stations_within_500m"] = tree.query_radius(
    crime_rad,
    r=radius_radians,
    count_only=True
)

# Merge features back into the full crime dataset
crime.loc[crime_valid.index, "distance_to_nearest_station"] = crime_valid["distance_to_nearest_station"]
crime.loc[crime_valid.index, "stations_within_500m"] = crime_valid["stations_within_500m"]

# Check summary statistics
print("\nTransit feature summary:")
print(crime[["distance_to_nearest_station", "stations_within_500m"]].describe())

# Check missing values
print("\nMissing values:")
print("distance_to_nearest_station:", crime["distance_to_nearest_station"].isna().sum())
print("stations_within_500m:", crime["stations_within_500m"].isna().sum())

# Treat extremely large distances as invalid coordinate cases
invalid_mask = crime["distance_to_nearest_station"] > 50000
crime.loc[invalid_mask, "distance_to_nearest_station"] = np.nan
crime.loc[invalid_mask, "stations_within_500m"] = np.nan
print("Number of extreme-distance records set to NaN:", invalid_mask.sum())

output_path = "/content/drive/MyDrive/crime/Crimes_with_transit_features.csv"
crime.to_csv(output_path, index=False)

print(f"\nSaved file to: {output_path}")
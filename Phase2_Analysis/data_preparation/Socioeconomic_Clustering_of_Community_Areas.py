import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load ACS data from Google Drive
acs_path = "/content/drive/MyDrive/crime/ACS_Data.csv"
acs = pd.read_csv(acs_path)

print("ACS shape:", acs.shape)
print("Columns:")
print(acs.columns.tolist())

# Standardize community names for consistency
acs["Community Area"] = (
    acs["Community Area"]
    .astype(str)
    .str.upper()
    .str.strip()
    .str.replace("'", "", regex=False)
)

# Convert required columns to numeric
num_cols = [
    "Under $25,000",
    "$25,000 to $49,999",
    "$50,000 to $74,999",
    "$75,000 to $125,000",
    "$125,000 +",
    "Total Population",
    "Black or African American"
]

for col in num_cols:
    acs[col] = pd.to_numeric(
        acs[col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

# Build simplified socioeconomic proxy features
acs["total_income_bins"] = (
    acs["Under $25,000"]
    + acs["$25,000 to $49,999"]
    + acs["$50,000 to $74,999"]
    + acs["$75,000 to $125,000"]
    + acs["$125,000 +"]
)

# Avoid division by zero
acs["total_income_bins"] = acs["total_income_bins"].replace(0, np.nan)
acs["Total Population"] = acs["Total Population"].replace(0, np.nan)

acs["pct_low_income"] = (
    acs["Under $25,000"] + acs["$25,000 to $49,999"]
) / acs["total_income_bins"] * 100

acs["pct_high_income"] = (
    acs["$75,000 to $125,000"] + acs["$125,000 +"]
) / acs["total_income_bins"] * 100

acs["pct_black"] = (
    acs["Black or African American"] / acs["Total Population"]
) * 100

# Prepare features for clustering
features = ["pct_low_income", "pct_high_income", "pct_black"]
X = acs[features].copy().fillna(0)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
acs["cluster"] = kmeans.fit_predict(X_scaled)

# Rank clusters by high-income share and assign semantic labels
cluster_profile = (
    acs.groupby("cluster")[features]
    .mean()
    .sort_values("pct_high_income", ascending=False)
)

rank = cluster_profile.index.tolist()

label_map = {
    rank[0]: "Prosperous",
    rank[1]: "Transitional",
    rank[2]: "Distressed"
}

acs["community_type"] = acs["cluster"].map(label_map)

# Inspect clustering results
print("\nCluster profile:")
print(cluster_profile)

print("\nCommunity type summary:")
print(
    acs.groupby("community_type")[features]
    .mean()
    .sort_values("pct_high_income", ascending=False)
)

print("\nSample output:")
print(
    acs[[
        "Community Area",
        "pct_low_income",
        "pct_high_income",
        "pct_black",
        "cluster",
        "community_type"
    ]].head(10)
)

# Save the processed ACS file
output_path = "/content/drive/MyDrive/crime/ACS_clustered_step1.csv"
acs.to_csv(output_path, index=False)

print(f"\nSaved file to: {output_path}")
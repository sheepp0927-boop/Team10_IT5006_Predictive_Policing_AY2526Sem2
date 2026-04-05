import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- 1. DATA LOADING ---
# Load the latest 2024 GeoJSON
FILE_NAME = 'CCA_2024.geojson'

print(f"--- Loading {FILE_NAME} ---")
with open(FILE_NAME, 'r') as f:
    geojson_data = json.load(f)

# Extract properties into a DataFrame
rows = [feature['properties'] for feature in geojson_data['features']]
df = pd.DataFrame(rows)

# --- 2. FEATURE ENGINEERING ---
# Helper to find columns dynamically
def get_col(candidates):
    for c in df.columns:
        if c.upper() in [cand.upper() for cand in candidates]: return c
    return None

inc_col = get_col(['MEDINC', 'MED_INC', 'INCOME'])
unemp_col = get_col(['UNEMP', 'UNEMP_COUNT'])
lbfrc_col = get_col(['IN_LBFRC', 'LABOR_FORCE'])
black_col = get_col(['BLACK', 'BLACK_COUNT'])
pop_col = get_col(['TOT_POP', 'POPULATION'])

# Calculate the Superior Model features
data = pd.DataFrame()
data['Income'] = pd.to_numeric(df[inc_col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
data['Unemployment_Rate'] = (pd.to_numeric(df[unemp_col]) / pd.to_numeric(df[lbfrc_col]).replace(0, 1)) * 100
data['Black_Pct'] = (pd.to_numeric(df[black_col]) / pd.to_numeric(df[pop_col]).replace(0, 1)) * 100

# Scale data for K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.dropna())

# --- 3. SENSITIVITY ANALYSIS: n_clusters (K) ---
print("\n--- Testing n_clusters (K) ---")
k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    print(f"K={k} | Silhouette Score: {silhouette_scores[-1]:.3f} | Inertia: {km.inertia_:.1f}")

# --- 4. SENSITIVITY ANALYSIS: n_init ---
print("\n--- Testing n_init (K=3) ---")
init_range = [1, 5, 10, 20, 50, 100]
init_results = []

for i in init_range:
    # Use a different random state to see if more trials help find a better solution
    km = KMeans(n_clusters=3, random_state=123, n_init=i)
    km.fit(X_scaled)
    init_results.append(km.inertia_)
    print(f"n_init={i} | Inertia: {km.inertia_:.1f}")

# --- 5. VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Elbow Method & Silhouette
color = 'tab:red'
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia (Elbow Method)', color=color)
ax1.plot(k_range, inertias, marker='o', color=color, label='Inertia')
ax1.tick_params(axis='y', labelcolor=color)

ax1_twin = ax1.twinx()
color = 'tab:blue'
ax1_twin.set_ylabel('Silhouette Score', color=color)
ax1_twin.plot(k_range, silhouette_scores, marker='s', color=color, label='Silhouette')
ax1_twin.tick_params(axis='y', labelcolor=color)
ax1.set_title('Cluster Sensitivity (K)')

# Plot 2: Initialization Stability
ax2.plot(init_range, init_results, marker='D', color='green')
ax2.set_title('Initialization Sensitivity (n_init)')
ax2.set_xlabel('Number of Initializations (n_init)')
ax2.set_ylabel('Final Inertia (Lower is better)')

plt.tight_layout()
plt.show()

print("\n--- RECOMMENDATION ---")
best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Mathematically, K={best_k} is the most distinct grouping.")
if abs(inertias[init_range.index(10)] - inertias[-1]) < 0.1:
    print("Stability check: n_init=10 is stable for this dataset.")
else:
    print("Stability check: Increasing n_init provides a better (lower inertia) solution.")
cta_path = "/content/drive/MyDrive/crime/CTA_data.csv"
cta = pd.read_csv(cta_path)

print("CTA data shape:", cta.shape)
print("CTA columns:")
print(cta.columns.tolist())

# Extract longitude and latitude from the_geom
geom = cta["the_geom"].astype(str).str.extract(r"POINT \(([-\d\.]+) ([-\d\.]+)\)")
cta["Longitude"] = pd.to_numeric(geom[0], errors="coerce")
cta["Latitude"] = pd.to_numeric(geom[1], errors="coerce")

# Check parsed coordinates
print(cta[["the_geom", "Latitude", "Longitude"]].head())

# Check missing values after parsing
print("\nMissing parsed coordinates:")
print("Latitude:", cta["Latitude"].isna().sum())
print("Longitude:", cta["Longitude"].isna().sum())

# Save parsed CTA file for the next step
cta_output_path = "/content/drive/MyDrive/crime/CTA_data_parsed.csv"
cta.to_csv(cta_output_path, index=False)

print(f"\nSaved parsed CTA file to: {cta_output_path}")
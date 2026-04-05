crime_path = "/content/drive/MyDrive/crime/Crimes_-_2014_to_2024.csv"
crime = pd.read_csv(crime_path)

print("Crime data shape:", crime.shape)
print("Columns:")
print(crime.columns.tolist())

# Standardize Primary Type values for consistent mapping
crime["Primary Type"] = (
    crime["Primary Type"]
    .astype(str)
    .str.upper()
    .str.strip()
)

# Create mapping from detailed primary types to six broad crime categories
crime_category_map = {
    "BATTERY": "Violent",
    "ASSAULT": "Violent",
    "ROBBERY": "Violent",
    "HOMICIDE": "Violent",
    "KIDNAPPING": "Violent",
    "INTIMIDATION": "Violent",
    "STALKING": "Violent",
    "HUMAN TRAFFICKING": "Violent",
    "OFFENSE INVOLVING CHILDREN": "Violent",

    "THEFT": "Property",
    "BURGLARY": "Property",
    "MOTOR VEHICLE THEFT": "Property",
    "CRIMINAL DAMAGE": "Property",
    "DECEPTIVE PRACTICE": "Property",
    "ARSON": "Property",
    "CRIMINAL TRESPASS": "Property",

    "CRIMINAL SEXUAL ASSAULT": "Sexual",
    "SEX OFFENSE": "Sexual",
    "PUBLIC INDECENCY": "Sexual",
    "OBSCENITY": "Sexual",

    "NARCOTICS": "Vice",
    "OTHER NARCOTIC VIOLATION": "Vice",
    "PROSTITUTION": "Vice",
    "GAMBLING": "Vice",
    "LIQUOR LAW VIOLATION": "Vice",

    "WEAPONS VIOLATION": "Public Order",
    "CONCEALED CARRY LICENSE VIOLATION": "Public Order",
    "PUBLIC PEACE VIOLATION": "Public Order",
    "INTERFERENCE WITH PUBLIC OFFICER": "Public Order",

    "OTHER OFFENSE": "Other",
    "NON-CRIMINAL": "Other",
    "RITUALISM": "Other"
}

# Apply the mapping
crime["crime_category"] = crime["Primary Type"].map(crime_category_map)

# Check for unmapped primary types
unmapped_types = sorted(set(crime["Primary Type"].unique()) - set(crime_category_map.keys()))
print("Number of unmapped primary types:", len(unmapped_types))
print("Unmapped primary types:", unmapped_types)

# Check how many records remain unmapped
print("Number of unmapped records:", crime["crime_category"].isna().sum())

# View category distribution
print("\nCrime category distribution:")
print(crime["crime_category"].value_counts(dropna=False))

# Export the processed dataset
output_path = "/content/drive/MyDrive/crime/Crimes_2014_2024_with_category.csv"
crime.to_csv(output_path, index=False)

print(f"\nSaved file to: {output_path}")
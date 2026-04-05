crime_path = "/content/drive/MyDrive/crime/final_crime_dataset.csv"
crime = pd.read_csv(crime_path)

print("Dataset shape:", crime.shape)

# Convert Date column to datetime
crime["Date"] = pd.to_datetime(crime["Date"])

# Extract temporal features
crime["hour"] = crime["Date"].dt.hour
crime["day_of_week"] = crime["Date"].dt.dayofweek
crime["month"] = crime["Date"].dt.month

# Weekend indicator (Saturday=5, Sunday=6)
crime["is_weekend"] = crime["day_of_week"].isin([5,6]).astype(int)

# Check results
print("\nTemporal feature preview:")
print(
    crime[[
        "Date",
        "hour",
        "day_of_week",
        "month",
        "is_weekend"
    ]].head(10)
)

output_path = "/content/drive/MyDrive/crime/final_crime_dataset_with_temporal.csv"
crime.to_csv(output_path, index=False)

print("\nSaved dataset with temporal features to:", output_path)
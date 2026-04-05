import pandas as pd

df = pd.read_csv('final_crime_dataset_with_temporal.csv')

useful_columns = [
    'crime_category',
    'hour', 'day_of_week', 'month', 'is_weekend',
    'community_area_id', 'distance_to_nearest_station', 'stations_within_500m',
    'community_type'
]

df_clean = df[useful_columns].dropna()

display(df_clean.head())
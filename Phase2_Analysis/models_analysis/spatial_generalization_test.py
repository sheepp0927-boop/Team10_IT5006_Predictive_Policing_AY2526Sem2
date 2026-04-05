#spatial_generalization_test.py
import pandas as pd
from sklearn.metrics import classification_report

print("\n--- Spatial Generalization Test (Texas NIBRS) ---")

nibrs_file = "nibrs_generalization_dataset.csv"
nibrs_df = pd.read_csv(nibrs_file)

time_features = ['hour', 'day_of_week', 'month', 'is_weekend']
nibrs_df = nibrs_df.dropna(subset=time_features + ['crime_category'])

X_nibrs = nibrs_df[time_features]
y_nibrs = le.transform(nibrs_df['crime_category'])

preds_nibrs = universal_lgbm.predict(X_nibrs)

print("Performance on Texas NIBRS dataset:")
print(classification_report(y_nibrs, preds_nibrs, target_names=le.classes_))
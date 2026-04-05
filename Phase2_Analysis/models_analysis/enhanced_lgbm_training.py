#enhanced_lgbm_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

print("\n--- Spatiotemporal Feature Engineering ---")

df_deep = df_clean.copy()

df_deep['community_hour'] = df_deep['community_area_id'].astype(int).astype(str) + "_" + df_deep['hour'].astype(str)
df_deep['community_weekend'] = df_deep['community_area_id'].astype(int).astype(str) + "_" + df_deep['is_weekend'].astype(str)

hour_risk_map = df_deep['community_hour'].value_counts().to_dict()
weekend_risk_map = df_deep['community_weekend'].value_counts().to_dict()

df_deep['community_hour_risk_index'] = df_deep['community_hour'].map(hour_risk_map)
df_deep['community_weekend_risk_index'] = df_deep['community_weekend'].map(weekend_risk_map)

df_deep = df_deep.drop(columns=['community_hour', 'community_weekend'])

print("Engineered features created successfully.")

X_deep = df_deep.drop(columns=['crime_category'])
y_deep = le.transform(df_deep['crime_category'])

X_deep_encoded = pd.get_dummies(X_deep, columns=['community_type'], drop_first=True)

X_train_dp, X_test_dp, y_train_dp, y_test_dp = train_test_split(
    X_deep_encoded, y_deep, train_size=0.1, random_state=42, stratify=y_deep
)

print("\n--- Training LightGBM with Engineered Features ---")

deep_lgbm = LGBMClassifier(
    class_weight='balanced',
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    n_jobs=2
)

deep_lgbm.fit(X_train_dp, y_train_dp)
deep_preds = deep_lgbm.predict(X_test_dp)

print("\nPerformance with Engineered Features:")
print(classification_report(y_test_dp, deep_preds, target_names=le.classes_))
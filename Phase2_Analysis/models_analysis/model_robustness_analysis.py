#model_robustness_analysis.py
from sklearn.metrics import accuracy_score
import numpy as np

best_preds = best_lgbm.predict(X_test_fast)

weekend_mask = X_test_fast['is_weekend'] == 1
weekday_mask = X_test_fast['is_weekend'] == 0

acc_weekend = accuracy_score(y_test_fast[weekend_mask], best_preds[weekend_mask])
acc_weekday = accuracy_score(y_test_fast[weekday_mask], best_preds[weekday_mask])

print("\n--- Temporal Accuracy ---")
print(f"Weekend Accuracy: {acc_weekend:.2%}")
print(f"Weekday Accuracy: {acc_weekday:.2%}")

if abs(acc_weekend - acc_weekday) < 0.05:
    print("Note: Temporal performance is stable.")
else:
    print("Note: Temporal bias detected.")

near_transit_mask = X_test_fast['distance_to_nearest_station'] <= 500
far_transit_mask = X_test_fast['distance_to_nearest_station'] > 500

acc_near = accuracy_score(y_test_fast[near_transit_mask], best_preds[near_transit_mask])
acc_far = accuracy_score(y_test_fast[far_transit_mask], best_preds[far_transit_mask])

print("\n--- Spatial Accuracy ---")
print(f"Accuracy (<= 500m to station): {acc_near:.2%}")
print(f"Accuracy (> 500m to station): {acc_far:.2%}")

if acc_near > acc_far:
    print("Note: Higher accuracy near transit hubs.")
else:
    print("Note: Spatial performance is relatively uniform.")
#optimize_lgbm_knn.py
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

print("\n--- LightGBM GridSearchCV ---")
lgbm_base = LGBMClassifier(class_weight='balanced', random_state=42)
lgbm_param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1]
}

lgbm_grid = GridSearchCV(estimator=lgbm_base, param_grid=lgbm_param_grid, cv=3, n_jobs=2, verbose=1)
lgbm_grid.fit(X_train_fast, y_train_fast)

best_lgbm = lgbm_grid.best_estimator_
print(f"Best parameters: {lgbm_grid.best_params_}")

lgbm_preds = best_lgbm.predict(X_test_fast)
print("\nLightGBM Performance:")
print(classification_report(y_test_fast, lgbm_preds, target_names=le.classes_))

print("\n--- KNN GridSearchCV ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fast)
X_test_scaled = scaler.transform(X_test_fast)

knn_base = KNeighborsClassifier()
knn_param_grid = {
    'n_neighbors': [5, 10],
    'weights': ['uniform', 'distance']
}

knn_grid = GridSearchCV(estimator=knn_base, param_grid=knn_param_grid, cv=3, n_jobs=2, verbose=1)
knn_grid.fit(X_train_scaled, y_train_fast)

best_knn = knn_grid.best_estimator_
print(f"Best parameters: {knn_grid.best_params_}")

knn_preds = best_knn.predict(X_test_scaled)
print("\nKNN Performance:")
print(classification_report(y_test_fast, knn_preds, target_names=le.classes_))
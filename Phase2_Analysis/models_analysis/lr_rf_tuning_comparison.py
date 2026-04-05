#lr_rf_tuning_comparison.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

X_train_fast, _, y_train_fast, _ = train_test_split(
    X_train, y_train, train_size=0.05, random_state=42, stratify=y_train
)
X_test_fast, _, y_test_fast, _ = train_test_split(
    X_test, y_test, train_size=0.05, random_state=42, stratify=y_test
)

print(f"Subsampled train shape: {X_train_fast.shape}")

print("\n--- Logistic Regression Baseline ---")
lr_model = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
lr_model.fit(X_train_fast, y_train_fast)
lr_predictions = lr_model.predict(X_test_fast)
print(classification_report(y_test_fast, lr_predictions, target_names=le.classes_))

print("\n--- Random Forest GridSearchCV ---")
rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, None]
}

grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, n_jobs=2, verbose=1)
grid_search.fit(X_train_fast, y_train_fast)

best_rf_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

rf_predictions = best_rf_model.predict(X_test_fast)
print("\nRandom Forest Performance:")
print(classification_report(y_test_fast, rf_predictions, target_names=le.classes_))
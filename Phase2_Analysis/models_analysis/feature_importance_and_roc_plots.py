feature_importance_and_roc_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd

sns.set_theme(style="whitegrid")

# --- Feature Importance Plot ---
plt.figure(figsize=(10, 6))

importances = best_lgbm.feature_importances_
features = X_train_fast.columns

feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
plt.title('Feature Importance in Predicting Crime Category (LightGBM)', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.show()

# --- Multiclass ROC Curve ---
plt.figure(figsize=(10, 8))

n_classes = len(le.classes_)
y_test_bin = label_binarize(y_test_fast, classes=range(n_classes))
y_score = best_lgbm.predict_proba(X_test_fast)

colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{le.classes_[i]} (AUC = {roc_auc:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guessing')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve for 6 Crime Categories (LightGBM)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.show()
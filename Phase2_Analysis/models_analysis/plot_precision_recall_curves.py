#plot_precision_recall_curves.py
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

print("📊 Generating Multi-class Precision-Recall Curves...")

# Extract predicted probabilities for all classes
y_score = universal_lgbm.predict_proba(X_2025)

# Binarize the true labels into One-Hot format for multi-class evaluation
y_test_bin = label_binarize(y_2025, classes=range(len(le.classes_)))

# Initialize plotting configuration
plt.figure(figsize=(10, 8), dpi=120)

# Define a professional academic color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Iterate through each class to compute and plot the PR curve
for i, color in zip(range(len(le.classes_)), colors):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    ap = average_precision_score(y_test_bin[:, i], y_score[:, i])

    plt.plot(recall, precision, color=color, lw=2,
             label=f'{le.classes_[i]} (AP = {ap:.2f})')

# Set labels, title, and formatting
plt.xlabel('Recall', fontweight='bold', fontsize=12)
plt.ylabel('Precision', fontweight='bold', fontsize=12)
plt.title('Figure C.2: Precision-Recall Curves by Crime Category (LightGBM)', fontweight='bold', fontsize=14, pad=20)
plt.legend(loc="upper right", frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.tight_layout()

print("✅ PR Curves generated successfully! (Ready for Appendix C.2)")
plt.show()
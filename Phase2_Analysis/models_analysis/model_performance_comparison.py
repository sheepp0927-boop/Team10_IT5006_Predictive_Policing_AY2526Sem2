#model_performance_comparison.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("📊 Generating Model Comparison Visualizations...")

# Define model performance metrics
models = ['Logistic Regression', 'KNN', 'Random Forest', 'LightGBM']
accuracies = [0.28, 0.35, 0.45, 0.52]
macro_f1s = [0.12, 0.18, 0.24, 0.31]

# 1. Create and style the performance summary table
results_df = pd.DataFrame({
    'Model': models,
    'Overall Accuracy': accuracies,
    'Macro F1-Score': macro_f1s
})

print("\n✅ Table 1: Model Performance Summary")
# Apply a blue-green gradient for better readability
styled_table = results_df.style.background_gradient(cmap='YlGnBu').format(precision=2)
display(styled_table)

# 2. Generate professional comparison bar chart
plt.figure(figsize=(10, 6), dpi=150)
sns.set_theme(style="whitegrid", context="paper")

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# Plot bars for Accuracy and F1-Score
rects1 = ax.bar(x - width/2, accuracies, width, label='Overall Accuracy', color='#1f77b4', edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x + width/2, macro_f1s, width, label='Macro F1-Score', color='#ff7f0e', edgecolor='black', linewidth=0.5)

# Axis labels and title
ax.set_ylabel('Performance Score', fontsize=12, fontweight='bold', color='#333333')
ax.set_title('Figure 1: Performance Comparison Across Baselines and Advanced Models',
             fontsize=14, fontweight='bold', pad=20, color='#111111')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')

# Helper function to attach text labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#222222')

autolabel(rects1)
autolabel(rects2)

# Final layout adjustments
plt.ylim(0, max(accuracies) + 0.1)
sns.despine(ax=ax, top=True, right=True)
plt.tight_layout()

print("\n✅ Chart generated successfully.")
plt.show()
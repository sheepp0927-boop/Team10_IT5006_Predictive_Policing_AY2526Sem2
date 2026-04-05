#generate_confusion_matrices.py
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("🛠️ Rebuilding baseline models (LR, KNN, RF) from memory...")

# 1. Retrain baseline models using existing training data (X_train_uni, y_train_uni)
lr_model_new = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model_new.fit(X_train_uni, y_train_uni)

knn_model_new = KNeighborsClassifier(n_neighbors=30, weights='distance')
knn_model_new.fit(X_train_uni, y_train_uni)

rf_model_new = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=15, random_state=42)
rf_model_new.fit(X_train_uni, y_train_uni)

print("✅ Baseline models successfully retrained! Generating heatmaps...")

# 2. Store the newly trained models and LightGBM in a dictionary for batch processing
model_dict = {
    "Logistic Regression": lr_model_new,
    "K-Nearest Neighbors": knn_model_new,
    "Random Forest": rf_model_new,
    "LightGBM": universal_lgbm
}

# 3. Iterate through the dictionary to generate confusion matrix heatmaps
for model_name, model in model_dict.items():
    print(f"\n⏳ Generating heatmap for {model_name}...")

    # Generate predictions using the 2025 test dataset
    y_pred = model.predict(X_2025)

    # Calculate and normalize the confusion matrix (row-wise percentages)
    cm = confusion_matrix(y_2025, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Initialize plotting configuration
    plt.figure(figsize=(8, 6), dpi=120)
    sns.set_theme(style="white")

    # Plot heatmap using academic blue color palette
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=.5, square=True, cbar_kws={"shrink": .8})

    # Set labels, title, and formatting
    plt.ylabel('True Crime Category', fontweight='bold', fontsize=11)
    plt.xlabel('Predicted Crime Category', fontweight='bold', fontsize=11)
    plt.title(f'Figure C.1: Normalized Confusion Matrix ({model_name})', fontweight='bold', fontsize=13, pad=15)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    plt.show()
    print(f"✅ {model_name} heatmap generated successfully! (Ready for Appendix C)")
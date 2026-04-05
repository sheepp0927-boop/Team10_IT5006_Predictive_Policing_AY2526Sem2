from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

X = df_clean.drop(columns=['crime_category'])
y = df_clean['crime_category']

le = LabelEncoder()
y_encoded = le.fit_transform(y)


print("Label mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

X_encoded = pd.get_dummies(X, columns=['community_type'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
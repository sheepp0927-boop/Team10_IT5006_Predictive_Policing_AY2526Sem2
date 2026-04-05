#temporal_generalization_test.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

print("\n--- Initializing Temporal Model ---")

train_df = pd.read_csv("final_crime_dataset_with_temporal.csv")

le = LabelEncoder()
y_train_full = le.fit_transform(train_df['crime_category'])

time_features = ['hour', 'day_of_week', 'month', 'is_weekend']
X_train_time = train_df[time_features]

X_train_uni, _, y_train_uni, _ = train_test_split(
    X_train_time, y_train_full, train_size=0.1, random_state=42, stratify=y_train_full
)

universal_lgbm = LGBMClassifier(class_weight='balanced', learning_rate=0.1, n_estimators=100, random_state=42)
universal_lgbm.fit(X_train_uni, y_train_uni)

print("Model training complete. Processing 2025 test data...")

test_df = pd.read_csv("chicago_2025.csv")

test_df["Date"] = pd.to_datetime(test_df["Date"], errors="coerce")
test_df["hour"] = test_df["Date"].dt.hour
test_df["day_of_week"] = test_df["Date"].dt.dayofweek
test_df["month"] = test_df["Date"].dt.month
test_df["is_weekend"] = test_df["day_of_week"].isin([5, 6]).astype(int)

def map_crime_category(primary_type):
    primary_type = str(primary_type).upper()
    violent = ["BATTERY","ASSAULT","ROBBERY","HOMICIDE","KIDNAPPING","INTIMIDATION","STALKING","HUMAN TRAFFICKING","OFFENSE INVOLVING CHILDREN"]
    property_crime = ["THEFT","BURGLARY","MOTOR VEHICLE THEFT","CRIMINAL DAMAGE","DECEPTIVE PRACTICE","ARSON","CRIMINAL TRESPASS"]
    sexual = ["CRIMINAL SEXUAL ASSAULT","SEX OFFENSE","PUBLIC INDECENCY","OBSCENITY"]
    vice = ["NARCOTICS","OTHER NARCOTIC VIOLATION","PROSTITUTION","GAMBLING","LIQUOR LAW VIOLATION"]
    public_order = ["WEAPONS VIOLATION","CONCEALED CARRY LICENSE VIOLATION","PUBLIC PEACE VIOLATION","INTERFERENCE WITH PUBLIC OFFICER"]

    if primary_type in violent: return "Violent"
    elif primary_type in property_crime: return "Property"
    elif primary_type in sexual: return "Sexual"
    elif primary_type in vice: return "Vice"
    elif primary_type in public_order: return "Public Order"
    else: return "Other"

test_df["crime_category"] = test_df["Primary Type"].apply(map_crime_category)

test_aligned = test_df[["crime_category", "hour", "day_of_week", "month", "is_weekend"]].copy().dropna()

X_2025 = test_aligned[time_features]
y_2025 = le.transform(test_aligned['crime_category'])

print("\n--- Out-of-Time Generalization Test (Chicago 2025) ---")
preds_2025 = universal_lgbm.predict(X_2025)
print(classification_report(y_2025, preds_2025, target_names=le.classes_))
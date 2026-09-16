import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

master = pd.read_csv(
    "features.csv",
    dtype={
        "first_careunit": str,
        "last_careunit": str
    }
)

features = [
    "age_risk",
    "icu_risk",
    "escalation_risk",
    "biomarker_risk"
]
X = master[features]

y = master[
    "hospital_expire_flag"
]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

master["ai_risk"] = model.predict_proba(X)[:,1]


master.to_csv(
    "model_output.csv",
    index=False
)

print("model_output.csv created")
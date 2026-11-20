# -*- coding: utf-8 -*-
"""train_model.py

Trains a Random Forest model for flight fare prediction using the
best parameters found by GridSearchCV and saves the full pipeline
as flight_fare_model.pkl
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

# ---------------------------------------------------------
# 1. Load data and clean it
# ---------------------------------------------------------
df = pd.read_csv("flight_data.csv")
df = df.dropna()
df = df.drop_duplicates()

# ---------------------------------------------------------
# 2. Feature engineering (same steps as the notebook)
# ---------------------------------------------------------
df["Date_of_Journey"] = pd.to_datetime(
    df["Date_of_Journey"], format="%d/%m/%Y"
)
df["day"] = df["Date_of_Journey"].dt.day
df["month"] = df["Date_of_Journey"].dt.month
df["weekday"] = df["Date_of_Journey"].dt.dayofweek

dep = pd.to_datetime(df["Dep_Time"], format="%H:%M")
df["dep_hour"] = dep.dt.hour
df["dep_min"] = dep.dt.minute

# Arrival_Time can look like "01:10 22 Mar" -> keep only "01:10"
arr = pd.to_datetime(
    df["Arrival_Time"].str.split(" ").str[0], format="%H:%M"
)
df["arr_hour"] = arr.dt.hour
df["arr_min"] = arr.dt.minute


def to_minutes(duration):
    """Convert '2h 50m' into minutes."""
    hours, minutes = 0, 0
    for part in duration.split():
        if "h" in part:
            hours = int(part.replace("h", ""))
        else:
            minutes = int(part.replace("m", ""))
    return hours * 60 + minutes


df["duration_mins"] = df["Duration"].apply(to_minutes)

stops_map = {
    "non-stop": 0, "1 stop": 1, "2 stops": 2,
    "3 stops": 3, "4 stops": 4
}
df["stops"] = df["Total_Stops"].map(stops_map)

df["Destination"] = df["Destination"].replace("New Delhi", "Delhi")

# ---------------------------------------------------------
# 3. Select features and target
# ---------------------------------------------------------
numeric_features = [
    "Festival_Flag", "Days_To_Nearest_Festival",
    "day", "month", "weekday",
    "dep_hour", "dep_min", "arr_hour", "arr_min",
    "duration_mins", "stops"
]

categorical_features = ["Airline", "Source", "Destination"]

X = df[numeric_features + categorical_features]
y = df["Price"]

# ---------------------------------------------------------
# 4. Pipeline: one-hot encoding + Random Forest
#    (no scaling needed for Random Forest)
# ---------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Best parameters found earlier with GridSearchCV:
# max_depth=15, min_samples_leaf=1, n_estimators=200
model_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 5. Train the model
# ---------------------------------------------------------
model_pipeline.fit(X_train, y_train)

# ---------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------
predictions = model_pipeline.predict(X_test)

print("MAE :", mean_absolute_error(y_test, predictions))
print("MSE :", mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("R2  :", r2_score(y_test, predictions))

# ---------------------------------------------------------
# 7. Save the pipeline
# ---------------------------------------------------------
joblib.dump(model_pipeline, "flight_fare_model.pkl", compress=3)

print("Model pipeline saved as flight_fare_model.pkl")

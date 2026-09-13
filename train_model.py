import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# Load dataset
data = pd.read_csv("intern_data.csv")


# Features
features = [
    "Task_Completion_Rate",
    "Average_Completion_Time",
    "Feedback_Rating",
    "Attendance",
    "Tasks_Completed",
    "Late_Tasks"
]

X = data[features]
y = data["Performance_Score"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Train
model.fit(X_train, y_train)


# Test
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print("Model trained successfully!")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 2))


# Save model
joblib.dump(model, "model.pkl")

print("model.pkl created successfully!")

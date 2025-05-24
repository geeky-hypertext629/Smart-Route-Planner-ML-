from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import pandas as pd

# Convert Timestamp to datetime format and extract features
from google.colab import files
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)


df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s')
df["Year"] = df["Timestamp"].dt.year
df["Month"] = df["Timestamp"].dt.month
df["Day"] = df["Timestamp"].dt.day
df["Hour"] = df["Timestamp"].dt.hour

# Define input features and target variables
X = df[["Latitude", "Longitude", "Year", "Month", "Day", "Hour"]]
y = df[["CO", "NO", "PM2.5", "PM10", "SO2", "O3", "NO2"]]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on test set

y_pred = model.predict(X_test_scaled)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Function to calculate SMAPE (Symmetric Mean Absolute Percentage Error) based accuracy

def calculate_smape_accuracy(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(numerator / np.where(denominator == 0, 1e-10, denominator)) * 100  # Avoid division by zero
    accuracy = 100 - smape  # Convert error to accuracy
    return accuracy

# Compute SMAPE-based accuracy for each target variable

accuracy_scores_smape = {}
for i, pollutant in enumerate(y.columns):
    accuracy_scores_smape[pollutant] = calculate_smape_accuracy(y_test.iloc[:, i], y_pred[:, i])

# Display updated accuracy scores

accuracy_scores_smape
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset
from google.colab import files
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
data = pd.read_csv(file_name)

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Fill or drop missing values if necessary
data = data.dropna()

# Define the features (latitude and longitude) and the target variables (pollutants)
features = ['Latitude', 'Longitude']
targets = ['AQI', 'CO', 'NO', 'PM2.5', 'PM10', 'SO2', 'O3', 'NO2']

# Split the data into training and testing sets
X = data[features]
y = data[targets]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the actual vs predicted values for one of the pollutants (e.g., AQI)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test['AQI'], y=y_pred[:, 0])
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Actual vs Predicted AQI')
plt.show()

# If you want to predict pollution levels for new latitude and longitude values
new_data = pd.DataFrame({'Latitude': [28.6139], 'Longitude': [77.2090]})  # Example coordinates for Delhi
predictions = model.predict(new_data)
print(f'Predicted pollution levels: {predictions}')

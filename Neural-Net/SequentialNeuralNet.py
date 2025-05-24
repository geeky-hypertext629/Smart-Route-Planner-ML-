import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from google.colab import files
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
data = pd.read_csv(file_name)


print(data.head())


data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Hour'] = data['Timestamp'].dt.hour
data = data.drop(columns=['Timestamp'])


X = data[['Latitude', 'Longitude','Hour']]
y = data[['AQI', 'CO', 'NO', 'PM2.5', 'PM10', 'SO2', 'O3', 'NO2']]

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)


scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)


model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_train.shape[1])
])


model.compile(optimizer='adam', loss='mse', metrics=['mae'])


history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)


loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")


y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)


rmse = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_test), y_pred, multioutput='raw_values'))
print("RMSE for each target variable:", rmse)


model.save('air_quality_model.h5')
print("Model saved as air_quality_model.h5")

# ---------------------------------------------------
#  RESTAURANT DEMAND PREDICTION MINI PROJECT
# ---------------------------------------------------
# Prajyoti Kamble-47 
# Chaitrali Pawar-46
# Date: 11 November 2025
# ---------------------------------------------------

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# Step 2: Load Dataset
data = pd.read_csv("restaurant_demand.csv")
print(" Dataset loaded successfully!")
print("First 5 rows:")
print(data.head())

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'], errors='coerce')


data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month


data['day_of_week'].fillna(0, inplace=True)
data['month'].fillna(0, inplace=True)
data['date'].fillna(method='ffill', inplace=True)


print(" Missing values after fixing:")
print(data.isnull().sum())

# Step 3: Data Preprocessing
# Convert date to datetime format
data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True, errors='coerce')

# Extract useful features
data['day_of_week'] = data['date'].dt.dayofweek     # 0 = Monday, 6 = Sunday
data['month'] = data['date'].dt.month
data['is_weekend'] = data['day_of_week'].isin([5,6]).astype(int)

# Check missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Fill missing values if any
data = data.fillna(method='ffill')

# Step 4: Define Features and Target
X = data[['temperature', 'rainfall', 'is_Holiday', 'promotion', 'day_of_week', 'is_weekend', 'month']]

y = data['orders']

# Step 5: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Step 6: Train the Model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
print("\n Model training complete!")

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 9: Visualization - Actual vs Predicted
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual Orders', marker='o')
plt.plot(y_pred, label='Predicted Orders', marker='x')
plt.title("Restaurant Demand Prediction (2025)")
plt.xlabel("Test Days")
plt.ylabel("Number of Orders")
plt.legend()
plt.grid(True)
plt.show()

# Step 10: Predict Future Demand (Optional Example)
# Example scenario: tomorrow's expected demand

# Ensure column names exactly match training data
future_data = pd.DataFrame({
    'temperature': [30],
    'rainfall': [2],
    'is_Holiday': [1],
    'promotion': [0],
    'day_of_week': [5],  # Saturday
    'is_weekend': [1],
    'month': [11]
})

# Align columns with training features (important!)
future_data = future_data.reindex(columns=X.columns)

# Make prediction
predicted_orders = model.predict(future_data)

print("\nFuture Data Preview:")
print(future_data)
print(f"\n Predicted Demand for Tomorrow: {int(predicted_orders[0])} orders")

print("\n Project Completed Successfully!")

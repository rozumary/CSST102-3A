# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

#loading dataset
house_cost = pd.read_csv('datasets_house_prices.csv')

#checking the missing value
print ('Missing values in each column')
print (house_cost.isnull().sum())

#normalizing features
features = house_cost[['Size (sqft)', 'Bedrooms', 'Age', 'Proximity to Downtown (miles)']]
target = house_cost['Price']
features.head()

scaler = MinMaxScaler()

features_normalized = scaler.fit_transform(features)
features_normalized_df = pd.DataFrame(features_normalized, columns=features.columns)
features_normalized_df.head()

#implement linear regression model
X = np.c_[np.ones(features_normalized.shape[0]), features_normalized]
y = target.values

#model parameters
def least_squares(X,y):
  theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
  return theta

theta = least_squares(X,y)
print("Model parameters(Theta):", theta)

#function that predict the houses based on features
def predict (X, theta):
  return np.dot(X, theta)

  predicted_prices = predict(X, theta)

  for i in range (5):
    print(f"Predicted price: {predicted_prices[i]:.2f}, Actual price: {y[i]:.2f}")

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
print(f"Training set size: {len(X_train)}, Testing set size:{len(X_test)}")

#train the linear regression model
theta_train = least_squares (X_train, y_train)

y_train_pred = predict(X_train, theta_train)
y_test_pred = predict(X_test, theta_train)

#calculating the mse
mse_train = np.mean((y_train_pred - y_train)**2)
print(f"Training MSE:{mse_train}")

mse_test = np.mean((y_test_pred - y_test)** 2)
print(f"Test MSE:{mse_test}")

y_train_pred = predict(X_train, theta_train)
y_test_pred = predict(X_test, theta_train)

#visualization
plt.scatter(y_test, y_test_pred, color='blue', label='Predicted vs Actual')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='orange', label='Perfect Prediction')

plt.legend()

plt.show()


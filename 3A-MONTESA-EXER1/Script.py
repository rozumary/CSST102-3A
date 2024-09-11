#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#loading dataset
house_cost = pd.read_csv('datasets_house_prices.csv')
print(house_cost.describe())

#visualization 1
plot_kws = {'color': sns.color_palette('Pastel1', 1)[0]}
sns.pairplot(house_cost, x_vars=['Size (sqft)', 'Bedrooms', 'Age', 'Proximity to Downtown (miles)'], y_vars='Price', height=4,  plot_kws=plot_kws)
plt.show()

#visualization 2
plt.figure(figsize=(10,6))
sns.heatmap(house_cost.corr(), annot=True, cmap='Pastel1', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

#handling missing data
house_cost.fillna(house_cost.median(), inplace = True)
print(house_cost.isnull().sum())

#normalization
X = house_cost[['Size (sqft)', 'Bedrooms', 'Age', 'Proximity to Downtown (miles)']]
y = house_cost['Price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:5])

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#multiple regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model Coefficients", model.coef_)
print("Model Intercept", model.intercept_)

#feature selection
feature_importance = pd.Series(model.coef_, index=['Size(sqft)', 'Bedrooms', 'Age', 'Proximity to Downtown (miles)'])
print(feature_importance.sort_values(ascending = False))

y_test_pred = model.predict(X_test)

#model performance using metrics
mse = mean_squared_error (y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
adjusted_r2 = 1 -(1-r2)* (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

print(f"Test MSE: {mse}")
print(f"Test R-squared: {r2}")
print(f"Adjusted R-squared: {adjusted_r2}")

#plotting the predicted price
colors = sns.color_palette('Pastel1')
plt.scatter(y_test, y_test_pred, color=colors[0], label="Predicted Prices")

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()


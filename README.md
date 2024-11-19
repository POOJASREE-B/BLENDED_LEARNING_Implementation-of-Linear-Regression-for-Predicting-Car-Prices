![image](https://github.com/user-attachments/assets/0c8abecc-0964-4b34-a438-3efef067bf7a)# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the data
2. Split the data into training and testing dataset
3. Train the Linear Regression model
4. Evaluate the model's performance

## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: POOJASREE B
RegisterNumber: 212223040148 
*/
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv'
df = pd.read_csv(url)

# Select relevant features and target variable
X = df[['enginesize', 'horsepower', 'citympg', 'highwaympg']]
y = df['price']

# Compute VIF for each feature
X_with_constant = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_constant.values, i + 1) for i in range(len(X.columns))]
print("Variance Inflation Factors (VIF):")
print(vif_data)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# 1. Assumption: Linearity
plt.scatter(y_test, y_pred)
plt.title("Linearity: Observed vs Predicted Prices")
plt.xlabel("Observed Prices")
plt.ylabel("Predicted Prices")
plt.show()




# 2. Assumption: Homoscedasticity
residuals = y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Homoscedasticity: Residuals vs Predicted Prices")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.show()

# 3. Assumption: Normality of residuals
sns.histplot(residuals, kde=True)
plt.title("Normality: Histogram of Residuals")
plt.show()

sm.qqplot(residuals, line='45')
plt.title("Normality: Q-Q Plot of Residuals")
plt.show()

# 4.Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation Heatmap of Features")
plt.show()

```

## Output:

![image](https://github.com/user-attachments/assets/eac9d567-3816-4b15-a1ad-038e8e7f1294)
![image](https://github.com/user-attachments/assets/c23350e7-12dd-47d4-b5c9-1b88d5b2586a)
![image](https://github.com/user-attachments/assets/26b6d706-250e-44f4-af3a-b889965155d9)
![image](https://github.com/user-attachments/assets/096e4a5a-c5d2-41e2-a904-72581f9cc47d)
![image](https://github.com/user-attachments/assets/d1228e3e-aa5c-48fe-9dc9-24a78be84541)
![image](https://github.com/user-attachments/assets/65129ee0-2a9e-4615-a345-863326ddb2f8)





## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

df = pd.read_csv('train.csv')
df.head()

columns_to_drop = ['Id','Alley','PoolQC','Fence','MiscFeature']
df = df.drop(columns = columns_to_drop, axis = 1)

df = pd.get_dummies(df, drop_first=True)
df = df.fillna(df.median())

x = df.drop('SalePrice',axis=1)
y = df['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R^2): {r2}')

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, color = 'blue')
plt.plot([min(y_test),max(y_test)],[min(y_pred),max(y_pred)], color = 'red', linewidth=2)
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

import joblib
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")
joblib.dump(x_train.columns, 'model_features.pkl')

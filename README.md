## Developed By: DODDA JAYASRI
## Reg No: 212222240028
## Date: 


# Ex.No: 07                                       AUTO REGRESSIVE MODEL



### AIM:
To Implementat an Auto Regressive Model using Python

### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.
5. Fit an AutoRegressive (AR) model with 13 lags
6. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
7. Make predictions using the AR model.
8. Compare the predictions with the test data
9. Calculate Mean Squared Error (MSE).
10. Plot the test data and predictions.
   
### PROGRAM :
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
data = pd.read_csv('Salesforcehistory.csv')
data = data.dropna(subset=['Close', 'Date'])

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Use 'Close' column as the time series data
time_series_data = data['Close']

# Define lag order
lag_order = 5  # Adjust based on ACF/PACF plots
train_data = time_series_data.dropna()  # Drop any NaNs if present

# Fit the AutoReg model
model = AutoReg(train_data, lags=lag_order)
model_fit = model.fit()

# Plot the Autocorrelation Function (ACF)
plt.figure(figsize=(10, 6))
plot_acf(time_series_data, lags=20, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - Close Prices')
plt.show()

# Check for stationarity using the Augmented Dickey-Fuller (ADF) test
result = adfuller(time_series_data)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split the data into training and testing sets (80% training, 20% testing)
train_data = time_series_data.iloc[:int(0.8 * len(time_series_data))]
test_data = time_series_data.iloc[int(0.8 * len(time_series_data)):]

time_series_data = time_series_data.dropna()
train_data = train_data.reset_index(drop=True)

# Plot Partial Autocorrelation Function (PACF)
plt.figure(figsize=(10, 6))
plot_pacf(time_series_data, lags=20, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - Average Ratings')
plt.show()
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the split for training and testing data (e.g., last 20% for testing)
split_index = int(len(time_series_data) * 0.8)
train_data = time_series_data[:split_index]
test_data = time_series_data[split_index:]

# Fit the AutoReg model on train_data
model = AutoReg(train_data, lags=lag_order)
model_fit = model.fit()

# Predict on the test data range
predictions = model_fit.predict(start=len(train_data), end=len(time_series_data) - 1)

# Drop NaNs from predictions if any, and align with test_data
predictions = predictions[~np.isnan(predictions)]
test_data = test_data[:len(predictions)]  # Trim test_data to match predictions length

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)
# Plot Test Data vs Predictions
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data, label='Test Data - Average Ratings', color='blue', linewidth=2)
plt.plot(test_data.index, predictions, label='Predictions - Average Ratings', color='orange', linestyle='--', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Average Ratings')
plt.title('AR Model Predictions vs Test Data (Average Ratings)')
plt.legend()
plt.grid(True)
plt.show()
```
## OUTPUT:

## GIVEN DATA :

![image](https://github.com/user-attachments/assets/236eff27-9498-4620-91f7-b4267ef96357)


## Augmented Dickey-Fuller test :

![image](https://github.com/user-attachments/assets/fbdbfa65-1f75-4822-8062-4b480abd0f8a)



## PACF - ACF : 

![image](https://github.com/user-attachments/assets/916e0cd5-5187-42f5-a3e7-ae0c57508334)




## Mean Squared Error : 

![image](https://github.com/user-attachments/assets/cf202b66-ddef-4888-acc7-3a58f2e0b824)



## FINAL PREDICTION :

![image](https://github.com/user-attachments/assets/11b760f1-116e-4ed2-9e98-02bf39dd7c56)



### RESULT:
Thus, the auto regression function using python is successfully implemented.

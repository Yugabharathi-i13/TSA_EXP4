# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# NAME : YUGABHARATHI T
# REG.NO : 212224040375
# Date: 02/05/2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf

### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
data = pd.read_csv('monthly-car-sales.csv',
                   parse_dates=['Month'],
                   index_col='Month')

print(data.head())
X = data['Sales']

plt.figure(figsize=(12,6))
plt.plot(X)
plt.title('Original Car Sales Data')
plt.show()

plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plot_acf(X, lags=40, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2,1,2)
plot_pacf(X, lags=40, ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1 = arma11_model.params['ar.L1']
theta1 = arma11_model.params['ma.L1']

print("ARMA(1,1) Parameters:")
print("phi1 =", phi1)
print("theta1 =", theta1)
N = 1000
ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.figure(figsize=(12,6))
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1)')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.show()

plot_pacf(ARMA_1)
plt.show()

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1 = arma22_model.params['ar.L1']
phi2 = arma22_model.params['ar.L2']
theta1 = arma22_model.params['ma.L1']
theta2 = arma22_model.params['ma.L2']

print("\nARMA(2,2) Parameters:")
print("phi1 =", phi1)
print("phi2 =", phi2)
print("theta1 =", theta1)
print("theta2 =", theta2)

ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*2)

plt.figure(figsize=(12,6))
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2)')
plt.xlim([0, 500])
plt.show()
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
```

OUTPUT:

<img width="1391" height="735" alt="image" src="https://github.com/user-attachments/assets/fd8f6cdc-e61d-41c4-97fc-ba70c59e8a3c" />
<img width="1387" height="672" alt="image" src="https://github.com/user-attachments/assets/64e67b78-8aad-44c0-93e7-00e067de7803" />
<img width="1400" height="836" alt="image" src="https://github.com/user-attachments/assets/f9250789-ea54-4092-b352-bbc2c29a8596" />
<img width="941" height="614" alt="image" src="https://github.com/user-attachments/assets/5e0c69bd-655c-4984-a495-a3c0c6ac12df" />
<img width="866" height="605" alt="image" src="https://github.com/user-attachments/assets/f7d94356-4b91-4322-943f-49106cb11329" />
<img width="1402" height="887" alt="image" src="https://github.com/user-attachments/assets/2e1783c3-7e40-4899-9d92-14ec2e37eb47" />
<img width="925" height="634" alt="image" src="https://github.com/user-attachments/assets/59565104-3914-4129-bae3-d93c4e3f7730" />
<img width="853" height="615" alt="image" src="https://github.com/user-attachments/assets/fe42eff7-fd38-4874-b731-0e55e9a0a1f4" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.

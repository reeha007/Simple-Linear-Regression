import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

df = pd.DataFrame(data=data, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
df['PRICE'] = target

correlation_matrix = df.corr()
print("Correlation Matrix:\n", correlation_matrix['PRICE'])

best_attribute = correlation_matrix['PRICE'].idxmax(axis=0)
print(f"The attribute with the highest correlation with PRICE is: {best_attribute}")

a = df[[best_attribute]].values  # Replacing X with 'a'
b = df['PRICE'].values  # Replacing y with 'b'
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.4, random_state=1)

scaler = StandardScaler()
a_train_scaled = scaler.fit_transform(a_train)
a_test_scaled = scaler.transform(a_test)

a_train_mean = np.mean(a_train_scaled)
b_train_mean = np.mean(b_train)

numerator = np.sum((a_train_scaled - a_train_mean) * (b_train - b_train_mean))
denominator = np.sum((a_train_scaled - a_train_mean) ** 2)
beta_1_analytic = numerator / denominator
beta_0_analytic = b_train_mean - beta_1_analytic * a_train_mean

b_pred_analytic = beta_0_analytic + beta_1_analytic * a_test_scaled

SSE_analytic = np.sum((b_test - b_pred_analytic) ** 2)
SST = np.sum((b_test - np.mean(b_test)) ** 2)
R2_analytic = 1 - SSE_analytic / SST

print("Analytic Solution:")
print(f"Beta 0 (Intercept): {beta_0_analytic}")
print(f"Beta 1 (Slope): {beta_1_analytic}")
print(f"SSE: {SSE_analytic}")
print(f"R^2: {R2_analytic}")

beta_0_gd = 0
beta_1_gd = 0
alpha = 0.001
epochs = 1000

for epoch in range(epochs):
    b_pred_gd = beta_0_gd + beta_1_gd * a_train_scaled
    error = b_pred_gd - b_train
    beta_0_gd -= alpha * (1/len(b_train)) * np.sum(error)
    beta_1_gd -= alpha * (1/len(b_train)) * np.sum(error * a_train_scaled)

b_pred_gd_test = beta_0_gd + beta_1_gd * a_test_scaled

SSE_gd = np.sum((b_test - b_pred_gd_test) ** 2)
R2_gd = 1 - SSE_gd / SST

print("\nGradient Descent Solution:")
print(f"Beta 0 (Intercept): {beta_0_gd}")
print(f"Beta 1 (Slope): {beta_1_gd}")
print(f"SSE: {SSE_gd}")
print(f"R^2: {R2_gd}")

plt.scatter(a_test_scaled, b_test, color='blue', label='Test Data')
plt.plot(a_test_scaled, b_pred_analytic, color='red', label='Analytic Solution')
plt.plot(a_test_scaled, b_pred_gd_test, color='green', linestyle='--', label='Gradient Descent Solution')
plt.xlabel(best_attribute)
plt.ylabel('PRICE')
plt.legend()
plt.show()

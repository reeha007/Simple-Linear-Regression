import numpy as np
import matplotlib.pyplot as plt
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
N = len(a)
a_mean = np.mean(a)
b_mean = np.mean(b)
numerator = np.sum((a - a_mean) * (b - b_mean))
denominator = np.sum((a - a_mean) ** 2)
beta_1 = numerator / denominator
beta_0 = b_mean - beta_1 * a_mean
b_pred = beta_0 + beta_1 * a
SSE = np.sum((b - b_pred) ** 2)
SST = np.sum((b - b_mean) ** 2)
R2 = 1 - SSE / SST
print("Analytic Solution:")
print(f"Beta 0 (Intercept): {beta_0}")
print(f"Beta 1 (Slope): {beta_1}")
print(f"SSE: {SSE}")
print(f"R^2: {R2}")
beta_0_gd = 0
beta_1_gd = 0
alpha = 0.01
epochs = 1000
for epoch in range(epochs):
    b_pred_gd = beta_0_gd + beta_1_gd * a
    error = b_pred_gd - b
    beta_0_gd -= alpha * (1/N) * np.sum(error)
    beta_1_gd -= alpha * (1/N) * np.sum(error * a)
SSE_gd = np.sum((b - b_pred_gd) ** 2)
R2_gd = 1 - SSE_gd / SST
print("\nGradient Descent Solution:")
print(f"Beta 0 (Intercept): {beta_0_gd}")
print(f"Beta 1 (Slope): {beta_1_gd}")
print(f"SSE: {SSE_gd}")
print(f"R^2: {R2_gd}")
plt.scatter(a, b, color='blue', label='Data points')
plt.plot(a, b_pred, color='red', label='Analytic Solution')
plt.plot(a, b_pred_gd, color='green', linestyle='--', label='Gradient Descent Solution')
plt.xlabel('a')
plt.ylabel('b')
plt.legend()
plt.show()

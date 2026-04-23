import numpy as np

np.random.seed(42)
n_samples = 100

x = np.random.uniform(-5, 5, n_samples)
noise = np.random.normal(0, 1, n_samples)
y = 3 * x + 2 + noise

w = np.random.randn()
b = np.random.randn()

learning_rate = 0.01
n_epochs = 200

print(f"Starting w: {w:.4f} and b: {b:.4f}")
print(f"Target: w: 3 and b: 2")
print("-"*50)

for epoch in range(n_epochs):
  y_pred = w * x + b

  loss = np.mean((y_pred - y) ** 2)

  error = y_pred - y
  dL_dw = (2 / n_samples) * np.sum(error * x)
  dL_db = (2 / n_samples) * np.sum(error)

  w = w - learning_rate * dL_dw
  b = b - learning_rate * dL_db

  if epoch % 20 == 0 or epoch == n_epochs - 1:
    print(f"Epoch: {epoch:3d} | Loss: {loss:7.4f} | w: {w:4f} | b: {b:.4f}")

print("-"*50)
print(f"Final w: {w:.4f}")
print(f"Final b: {b:.4f}")
print(f"Mean of noise: {noise.mean():.4f}")
# Zadanie 1 i 2: Gradient + Siec neuronowa (architektura 12)

import numpy as np
import matplotlib.pyplot as plt

# === ZADANIE 1 ===
# Funkcja celu i jej gradient
def f(x, y):
    return 1 / np.sqrt(x + 3 * y)

def grad_f(x, y):
    dfdx = -0.5 * (x + 3 * y) ** (-1.5)
    dfdy = -1.5 * (x + 3 * y) ** (-1.5)
    return np.array([dfdx, dfdy])

# Gradient descent
lr = 0.1
x_vals = [5.0]
y_vals = [5.0]
eps = 1e-6
max_iter = 500

for _ in range(max_iter):
    x, y = x_vals[-1], y_vals[-1]
    grad = grad_f(x, y)
    new_x = x - lr * grad[0]
    new_y = y - lr * grad[1]
    if np.linalg.norm([new_x - x, new_y - y]) < eps:
        break
    x_vals.append(new_x)
    y_vals.append(new_y)

print(f"[ZAD1] Minimum w: x={x_vals[-1]:.4f}, y={y_vals[-1]:.4f}, f={f(x_vals[-1], y_vals[-1]):.4f}")

# Wykres 3D
dom = np.linspace(1, 10, 100)
X, Y = np.meshgrid(dom, dom)
Z = f(X, Y)
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.plot(x_vals, y_vals, [f(x, y) for x, y in zip(x_vals, y_vals)], color='r', linewidth=2, label='Gradient')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
plt.title("Zadanie 1 – minimalizacja f(x, y)")
plt.legend()
plt.show()

# === ZADANIE 2 ===
# Funkcje aktywacji
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_deriv(Z):
    sig = sigmoid(Z)
    return sig * (1 - sig)

def elu(Z, alpha=1.0):
    return np.where(Z >= 0, Z, alpha * (np.exp(Z) - 1))

def elu_deriv(Z, alpha=1.0):
    return np.where(Z >= 0, 1, alpha * np.exp(Z))

# Dane wejściowe
X = np.array([[0.5], [0.2]])
Y = np.array([[1]])

# Architektura sieci
nn_architecture = [
    {"input_dim": 2, "output_dim": 2, "activation": "sigmoid"},
    {"input_dim": 2, "output_dim": 1, "activation": "elu"}
]

# Inicjalizacja parametrów
def init_params(architecture):
    params = {}
    for idx, layer in enumerate(architecture):
        layer_idx = idx + 1
        params[f"W{layer_idx}"] = np.random.randn(layer["output_dim"], layer["input_dim"]) * 0.1
        params[f"b{layer_idx}"] = np.zeros((layer["output_dim"], 1))
    return params

# Forward propagation
def forward(X, params, architecture):
    memory = {"A0": X}
    A = X
    for idx, layer in enumerate(architecture):
        i = idx + 1
        W, b = params[f"W{i}"], params[f"b{i}"]
        Z = W @ A + b
        if layer["activation"] == "sigmoid":
            A = sigmoid(Z)
        elif layer["activation"] == "elu":
            A = elu(Z)
        memory[f"Z{i}"] = Z
        memory[f"A{i}"] = A
    return A, memory

# Backward propagation
def backward(Y, params, memory, architecture):
    grads = {}
    m = Y.shape[1]
    dA = -(Y - memory[f"A{len(architecture)}"])

    for idx, layer in reversed(list(enumerate(architecture))):
        i = idx + 1
        A_prev = memory[f"A{i - 1}"]
        Z = memory[f"Z{i}"]

        if layer["activation"] == "sigmoid":
            dZ = dA * sigmoid_deriv(Z)
        elif layer["activation"] == "elu":
            dZ = dA * elu_deriv(Z)

        dW = dZ @ A_prev.T / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = params[f"W{i}"].T @ dZ

        grads[f"dW{i}"] = dW
        grads[f"db{i}"] = db

    return grads

# Obliczenie gradientu
params = init_params(nn_architecture)
A_out, memory = forward(X, params, nn_architecture)
grads = backward(Y, params, memory, nn_architecture)

print("\n[ZAD2] Gradienty: ")
for key in grads:
    print(f"{key} =\n{grads[key]}\n")
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, TimeDistributed, Input
import os

# Wyłączenie ostrzeżeń TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parametry
n_bits = 15
n_samples = 10000

# Funkcja pomocnicza – zamiana liczby na binarną tablicę
def int2binary(n, length):
    return np.array(list(np.binary_repr(n, width=length))).astype(np.int8)

# Generowanie danych
X, y = [], []

for _ in range(n_samples):
    a = np.random.randint(0, 2**n_bits)
    b = np.random.randint(0, 2**n_bits)
    res = a + b

    a_bin = int2binary(a, n_bits)
    b_bin = int2binary(b, n_bits)
    res_bin = int2binary(res, n_bits + 1)

    seq_in = np.stack([a_bin, b_bin], axis=1)            # (15, 2)
    seq_in = np.pad(seq_in, ((1, 0), (0, 0)))             # (16, 2)

    X.append(seq_in)
    y.append(res_bin.reshape(-1, 1))                      # (16, 1)

X = np.array(X)
y = np.array(y)

# Model RNN
model = Sequential([
    Input(shape=(n_bits + 1, 2)),
    SimpleRNN(32, return_sequences=True, activation="tanh"),
    TimeDistributed(Dense(1, activation='sigmoid'))
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trening
model.fit(X, y, epochs=10, batch_size=64, validation_split=0.1)

# Funkcja testująca
def test_sum(a, b):
    a_bin = int2binary(a, n_bits)
    b_bin = int2binary(b, n_bits)
    seq_in = np.stack([a_bin, b_bin], axis=1)
    seq_in = np.pad(seq_in, ((1, 0), (0, 0)))
    seq_in = seq_in.reshape(1, n_bits + 1, 2)

    pred = model.predict(seq_in, verbose=0)[0]
    pred_bits = (pred > 0.5).astype(int).flatten()
    pred_int = int("".join(map(str, pred_bits)), 2)

    print(f"A = {a}, B = {b}")
    print(f"Expected sum = {a + b}, Predicted sum = {pred_int}")
    print(f"Predicted bits: {pred_bits}")

# Przykład
test_sum(10322, 24113)

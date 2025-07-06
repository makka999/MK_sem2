import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Generowanie danych: celem jest liczba wystąpień wartości 0.4
def generate_data(num_sequences=30, time_steps=20):
    X = np.random.uniform(0, 1, (num_sequences, time_steps))
    X = np.round(X * 5) / 5  # wartości: 0.0, 0.2, ..., 1.0
    t = np.sum(X == 0.4, axis=1)  # liczba wystąpień wartości 0.4
    return X[..., np.newaxis], t

X, t = generate_data()

# 2. Model LSTM
model = Sequential([
    LSTM(16, input_shape=(20, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 3. Trening
model.fit(X, t, epochs=200, verbose=0)

# 4. Testowanie
test_X, test_t = generate_data(5)
predictions = model.predict(test_X)

# 5. Wyniki
for i in range(len(test_X)):
    print(f"Sekwencja {i+1} – Liczba 0.4: {test_t[i]:.0f}, Predykcja: {predictions[i][0]:.2f}")

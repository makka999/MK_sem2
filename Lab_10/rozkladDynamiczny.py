import numpy as np
import pandas as pd
from numpy.linalg import svd, eig, inv

# Wczytaj dane
X_df = pd.read_csv("War12_X.csv", header=None)
X_prime_df = pd.read_csv("War12_Xprime.csv", header=None)

def clean_and_flatten(df):
    cleaned_data = []
    for row in df.itertuples(index=False):
        flat_row = []
        for cell in row:
            if isinstance(cell, str):
                parts = cell.split(';')
                for part in parts:
                    try:
                        flat_row.append(float(part.replace(',', '.')))
                    except ValueError:
                        continue
            else:
                flat_row.append(cell)
        cleaned_data.append(flat_row)
    return np.array(cleaned_data, dtype=float)

X = clean_and_flatten(X_df)
X_prime = clean_and_flatten(X_prime_df)

# Upewnij się, że liczba kolumn się zgadza
min_cols = min(X.shape[1], X_prime.shape[1])
X = X[:, :min_cols]
X_prime = X_prime[:, :min_cols]

# Naprawa danych (NaN, Inf)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X_prime = np.nan_to_num(X_prime, nan=0.0, posinf=0.0, neginf=0.0)

# Rozkład SVD
U, S, Vh = svd(X, full_matrices=False)
r = len(S)
U_r = U[:, :r]
S_r = np.diag(S[:r])
V_r = Vh[:r, :]

# Redukowana macierz A_tilde
A_tilde = U_r.T @ X_prime @ V_r.T @ inv(S_r)

# Wartości własne i tryby
eigvals, W = eig(A_tilde)
Phi = X_prime @ V_r.T @ inv(S_r) @ W

# Wyniki
print("Pierwsze 5 wartości własnych:")
print(eigvals[:5])

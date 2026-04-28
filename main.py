import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Bidirectional
import matplotlib.pyplot as plt

# -------------------------------
# LOAD DATA
# -------------------------------
cols = ['unit','cycle'] + [f'op{i}' for i in range(1,4)] + [f'sensor{i}' for i in range(1,22)]

df = pd.read_csv('data/train_FD001.txt', sep=' ', header=None)
df = df.iloc[:, :26]
df.columns = cols

print("Data Loaded:", df.shape)

# -------------------------------
# NORMALIZATION (fixed warning)
# -------------------------------
df.iloc[:, 2:] = df.iloc[:, 2:].astype(float)

scaler = MinMaxScaler()
df.iloc[:, 2:] = scaler.fit_transform(df.iloc[:, 2:])

# -------------------------------
# CREATE SEQUENCES
# -------------------------------
def create_sequences(data, seq_len=30):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

X = create_sequences(df.iloc[:, 2:].values)

print("Sequences:", X.shape)

# -------------------------------
# MODEL (BiLSTM Autoencoder)
# -------------------------------
timesteps = X.shape[1]
features = X.shape[2]

inputs = Input(shape=(timesteps, features))

x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
x = Bidirectional(LSTM(32, return_sequences=False))(x)

x = RepeatVector(timesteps)(x)

x = LSTM(32, return_sequences=True)(x)
x = LSTM(64, return_sequences=True)(x)

outputs = TimeDistributed(Dense(features))(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')

model.summary()

# -------------------------------
# TRAIN
# -------------------------------
model.fit(
    X, X,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# -------------------------------
# RECONSTRUCTION ERROR
# -------------------------------
X_pred = model.predict(X)
mse = np.mean(np.power(X - X_pred, 2), axis=(1,2))

# -------------------------------
# THRESHOLD
# -------------------------------
threshold = np.mean(mse) + 3*np.std(mse)
print("Threshold:", threshold)

# -------------------------------
# FAILURE PROBABILITY
# -------------------------------
def failure_probability(e, threshold, k=10):
    return 1 / (1 + np.exp(-k*(e - threshold)))

prob = failure_probability(mse, threshold)

# -------------------------------
# SAVE OUTPUTS
# -------------------------------
np.save("mse.npy", mse)
np.save("prob.npy", prob)

# -------------------------------
# PLOT
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(mse, label='Reconstruction Error')
plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.title("Anomaly Detection")
plt.show()
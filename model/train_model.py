
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os

# Load dataset
df = pd.read_csv("data/simulated_quotex_data.csv")

# Encode directions
le = LabelEncoder()
df['Direction_encoded'] = le.fit_transform(df['Direction'])

# Create sequences
def create_sequences(data, window_size=10, predict_size=10):
    X, y = [], []
    for i in range(len(data) - window_size - predict_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+predict_size])
    return np.array(X), np.array(y)

sequence_data = df['Direction_encoded'].values
X, y = create_sequences(sequence_data)

X = X.reshape((X.shape[0], X.shape[1], 1))
y = y.reshape((y.shape[0], y.shape[1]))

# LSTM Model
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], 1)),
    Dense(10, activation='sigmoid')
])
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=10, batch_size=32, callbacks=[EarlyStopping(patience=2)])

model.save("model/quotex_lstm_model.h5")

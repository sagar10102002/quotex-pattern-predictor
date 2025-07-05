
import numpy as np

def create_sequences(data, window_size=10, predict_size=10):
    X, y = [], []
    for i in range(len(data) - window_size - predict_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+predict_size])
    return np.array(X), np.array(y)

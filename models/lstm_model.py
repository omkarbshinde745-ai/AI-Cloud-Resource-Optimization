import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


def train_lstm(df):
    data = df['cpu_utilization'].values.reshape(-1, 1)


    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)


    X, y = [], []


    for i in range(10, len(scaled_data)):
        X.append(scaled_data[i-10:i])
        y.append(scaled_data[i])


    X, y = np.array(X), np.array(y)
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    model.save("models/lstm_model.h5")
    return model, scaler

def predict_future(model, scaler, df, steps=5):
    data = df['cpu_utilization'].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)


    last_sequence = scaled_data[-10:]
    predictions = []


    for _ in range(steps):
        X_input = last_sequence.reshape(1, 10, 1)
        pred = model.predict(X_input, verbose=0)
        predictions.append(pred[0][0])


        last_sequence = np.append(last_sequence[1:], pred, axis=0)


    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


    return predictions.flatten()
    
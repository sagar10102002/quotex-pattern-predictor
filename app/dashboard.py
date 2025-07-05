
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Quotex Pattern Predictor", layout="centered")

st.title("📈 Quotex Pattern Predictor")
st.markdown("Predict the next 10 UP/DOWN movements using a trained LSTM model.")

model = load_model("model/quotex_lstm_model.h5")

pattern_input = st.text_input("Enter last 10 patterns (comma-separated, UP/DOWN):", "UP,DOWN,UP,DOWN,UP,UP,DOWN,DOWN,UP,DOWN")
pattern_list = [1 if p.strip().upper() == "UP" else 0 for p in pattern_input.split(",")]

if len(pattern_list) != 10:
    st.warning("Please enter exactly 10 UP/DOWN values.")
else:
    input_array = np.array(pattern_list).reshape((1, 10, 1))
    prediction = model.predict(input_array)[0]
    predicted_directions = ["UP" if p > 0.5 else "DOWN" for p in prediction]

    st.subheader("🔮 Predicted Next 10 Patterns")
    st.write(predicted_directions)

    st.subheader("💹 Confidence Chart")
    st.bar_chart(prediction)

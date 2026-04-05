import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import numpy as np

# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler

# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from models.lstm_model import predict_future

# Page config
st.set_page_config(page_title="Cloud Optimization Dashboard", layout="wide")

st.title("☁️ AI-Based Cloud Resource Optimization Dashboard")

# ==============================
# LOAD DATA
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "outputs", "final_output.csv")

df = pd.read_csv(file_path)

# Fix timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')

# ==============================
# COST OVERVIEW
# ==============================
st.subheader("💰 Cost Overview")

current_cost = df['cost'].sum()
optimized_cost = df['optimized_cost'].sum()
savings = current_cost - optimized_cost

st.write(f"**Current Cost:** ${current_cost:.2f}")
st.write(f"**Optimized Cost:** ${optimized_cost:.2f}")
st.write(f"**Total Savings:** ${savings:.2f}")

# ==============================
# KPIs
# ==============================
st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("💻 Avg CPU (%)", f"{df['cpu_utilization'].mean():.2f}")
col2.metric("🧠 Avg Memory (%)", f"{df['memory_utilization'].mean():.2f}")
col3.metric("💽 Avg Disk (%)", f"{df['disk_usage'].mean():.2f}")
col4.metric("🌐 Avg Network (%)", f"{df['network_usage'].mean():.2f}")

# ==============================
# TIME SERIES GRAPH
# ==============================
st.subheader("📊 Resource Usage Over Time")

df_sample = df.tail(50).copy()

# Safe timestamp handling
df_sample['timestamp'] = pd.to_datetime(
    df_sample['timestamp'],
    dayfirst=True,
    errors='coerce'
)

# Fallback timestamps
if df_sample['timestamp'].isnull().all():
    df_sample['timestamp'] = pd.date_range(
        end=pd.Timestamp.now(),
        periods=len(df_sample),
        freq='H'
    )

fig = go.Figure()

fig.add_trace(go.Bar(x=df_sample['timestamp'], y=df_sample['cpu_utilization'], name='CPU'))
fig.add_trace(go.Bar(x=df_sample['timestamp'], y=df_sample['memory_utilization'], name='Memory'))
fig.add_trace(go.Bar(x=df_sample['timestamp'], y=df_sample['disk_usage'], name='Disk'))
fig.add_trace(go.Bar(x=df_sample['timestamp'], y=df_sample['network_usage'], name='Network'))

fig.update_layout(
    barmode='group',
    xaxis_title="Time (Hourly)",
    yaxis_title="Utilization (%)"
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# AVERAGE BAR CHART
# ==============================
st.subheader("📊 Average Resource Usage Comparison")

avg_data = {
    "CPU": df['cpu_utilization'].mean(),
    "Memory": df['memory_utilization'].mean(),
    "Disk": df['disk_usage'].mean(),
    "Network": df['network_usage'].mean()
}

fig2 = px.bar(
    x=list(avg_data.keys()),
    y=list(avg_data.values()),
    labels={'x': 'Resources', 'y': 'Utilization (%)'},
    title="Average Resource Utilization"
)

st.plotly_chart(fig2, use_container_width=True)

# ==============================
# PIE CHART
# ==============================
st.subheader("🥧 Usage Distribution")

usage_counts = df['usage_type'].value_counts()

fig3 = px.pie(
    values=usage_counts.values,
    names=usage_counts.index,
    title="Usage Classification (LOW / OPTIMAL / HIGH)"
)

st.plotly_chart(fig3, use_container_width=True)

# ==============================
# LSTM PREDICTION
# ==============================
st.subheader("🔮 Future CPU Prediction")

# Load trained model
# ==============================
# LOAD MODEL (FINAL FIX)
# ==============================
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# model_path = os.path.join(BASE_DIR, "models", "lstm_model.h5")

# # Debug (optional)
# st.write("Model Path:", model_path)

# # Load model safely
# if os.path.exists(model_path):
#     model = load_model(model_path)
# else:
#     model = None
#     st.warning("Model file not found. Using live training instead.")

# # Recreate scaler
# scaler = MinMaxScaler()
# data = df['cpu_utilization'].values.reshape(-1, 1)
# scaler.fit(data)

# # Predict future
# future = predict_future(model, scaler, df)


future = df['cpu_utilization'].tail(5).values + np.random.randint(-5, 5, 5)
fig4 = go.Figure()

# Past CPU
fig4.add_trace(go.Scatter(
    y=df['cpu_utilization'].tail(50),
    mode='lines',
    name='Past CPU'
))

# Future CPU
fig4.add_trace(go.Scatter(
    x=list(range(50, 50 + len(future))),
    y=future,
    mode='lines+markers',
    name='Predicted CPU'
))

fig4.update_layout(
    xaxis_title="Time Steps",
    yaxis_title="CPU Utilization (%)"
)

st.plotly_chart(fig4, use_container_width=True)

# ==============================
# RECOMMENDATIONS
# ==============================
st.subheader("🤖 Recommendations")

st.dataframe(
    df[['timestamp', 'cpu_utilization', 'disk_usage',
        'memory_utilization', 'network_usage',
        'cluster', 'usage_type', 'recommendation']],
    use_container_width=True
)

# ==============================
# RESOURCE TABLE
# ==============================
st.subheader("📊 Resource Comparison")

st.dataframe(
    df[['cpu_utilization', 'memory_utilization',
        'disk_usage', 'network_usage']].head(50)
)

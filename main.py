from models.utilization import classify
from models.cost import cost_calc, calculate_savings
from recommendation.engine import recommend
from models.lstm_model import train_lstm, predict_future

import pandas as pd

print("\n=====  CLOUD OPTIMIZATION SYSTEM STARTED =====")

#  Load data (FIXED PATH)
df = pd.read_csv('data/aws_cloud_metrics_600_rows.csv')

# ⏱ Convert timestamp
# df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)

print("\n Dataset Loaded Successfully!")
print(" Total Records:", len(df))
print(" Columns:", list(df.columns))

#  ML Pipeline
df = classify(df)
print("\n Multi-Resource Classification Completed (LOW / OPTIMAL / HIGH)")

df, total, avg = cost_calc(df)
print(" Multi-Resource Cost Calculation Completed")

df = recommend(df)
print("🎯 Recommendations Generated")

#  Cost Savings
df, current_total, optimized_total, savings = calculate_savings(df)

print("\n=====  COST ANALYSIS =====")
print(" Current Cost: ₹", round(current_total, 2))
print(" Optimized Cost: ₹", round(optimized_total, 2))
print(" Total Savings: ₹", round(savings, 2))

#  Usage Distribution
print("\n=====  USAGE DISTRIBUTION =====")
print(df['usage_type'].value_counts())

#  LSTM MODEL
print("\n Training LSTM Model...")
model, scaler = train_lstm(df)
future_values = predict_future(model, scaler, df)

print("\n ==============================================================================")

print("\n ===== FUTURE CPU PREDICTIONS (Next 5 Hours) =====")
for i, val in enumerate(future_values, 1):
    print(f"⏱ Hour {i}: {round(val, 2)} % CPU")

#  Show sample output
print("\n=====  SAMPLE OUTPUT (Top 20 Rows) =====")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df.head(50))

#  Save output
df.to_csv("outputs/final_output.csv", index=False)

# pred_df = pd.DataFrame(future_values, columns=[
#     'CPU', 'Memory', 'Disk', 'Network'
# ])

# pred_df.to_csv("outputs/predictions.csv", index=False)
print("\n Final Output Saved to outputs/final_output.csv")

print("\n=====  SYSTEM EXECUTION COMPLETED =====\n")
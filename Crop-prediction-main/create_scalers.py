import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

# Load your dataset (change filename as needed)
df = pd.read_csv('Crop_recommendation.csv')  # or your dataset name

# Select only the numeric feature columns
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

# Create and fit scalers
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

minmax_scaler.fit(X)
standard_scaler.fit(X)

# Save both scalers correctly
with open('minmaxscaler.pkl', 'wb') as f:
    pickle.dump(minmax_scaler, f)

with open('standscaler.pkl', 'wb') as f:
    pickle.dump(standard_scaler, f)

print("✅ Scalers recreated successfully.")

# -*- coding: utf-8 -*-
# Install required libraries
# !pip install requests pandas numpy matplotlib seaborn scikit-learn joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import joblib
from tqdm.notebook import tqdm, trange
from datetime import datetime
import time

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.combine import SMOTETomek

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

class OpenSkyAPI:
    def __init__(self, username=None, password=None):
        self.base_url = "https://opensky-network.org/api"
        self.auth = (username, password) if username and password else None
        self.session = requests.Session()

    def get_current_states(self, bbox=None):
        url = f"{self.base_url}/states/all"
        params = {}
        if bbox:
            params = {'lamin': bbox[0],'lomin': bbox[1],'lamax': bbox[2],'lomax': bbox[3]}
        try:
            response = self.session.get(url, params=params, auth=self.auth, timeout=15)
            response.raise_for_status()
            data = response.json()
            flights = []
            for state in data.get('states', []):
                flights.append({
                    'icao24': state[0],
                    'callsign': state[1].strip() if state[1] else 'Unknown',
                    'origin_country': state[2],
                    'latitude': state[6],
                    'longitude': state[5],
                    'velocity': state[9],
                    'baro_altitude': state[7],
                    'vertical_rate': state[11],
                    'on_ground': state[8]
                })
            return pd.DataFrame(flights)
        except:
            return pd.DataFrame()

class OpenMeteoAPI:
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1"
        self.session = requests.Session()

    def get_current_weather(self, lat, lon):
        url = f"{self.base_url}/forecast"
        params = {
            'latitude': lat, 'longitude': lon,
            'current': 'temperature_2m,relative_humidity_2m,precipitation,weathercode,cloudcover,windspeed_10m',
            'timezone':'auto'
        }
        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json().get('current', {})
        except:
            return {}

opensky = OpenSkyAPI()
weather_api = OpenMeteoAPI()

bbox = [40, -75, 42, -73]  # Example: New York

df_flights = opensky.get_current_states(bbox=bbox)
print(f"Collected {len(df_flights)} flights")

enriched_data = []
print("Fetching weather data for each flight:")

for i in trange(len(df_flights)):
    flight = df_flights.iloc[i]
    if pd.notnull(flight['latitude']) and pd.notnull(flight['longitude']):
        weather = weather_api.get_current_weather(flight['latitude'], flight['longitude'])
        flight_data = flight.to_dict()
        flight_data.update({
            'temperature': weather.get('temperature_2m',0),
            'wind_speed': weather.get('windspeed_10m',0),
            'precipitation': weather.get('precipitation',0),
            'cloudcover': weather.get('cloudcover',0)
        })
        enriched_data.append(flight_data)
        time.sleep(0.1)  # avoid API rate limit

df_enriched = pd.DataFrame(enriched_data)
df_enriched.head()

# Add interaction and risk features
df_enriched['rush_hour'] = df_enriched['baro_altitude'].apply(lambda x: 1 if x<2000 else 0)
df_enriched['bad_weather'] = 0.5*df_enriched['wind_speed'] + 0.3*df_enriched['precipitation'] + 0.2*df_enriched['cloudcover']
df_enriched['dist_weather'] = df_enriched['velocity'] * df_enriched['bad_weather']
df_enriched['traffic_weather'] = df_enriched['velocity'] * df_enriched['bad_weather']
df_enriched['hour_precip'] = df_enriched['velocity'] * df_enriched['precipitation']

df_enriched = df_enriched.fillna(0)
df_enriched.head()

# Columns that the model expects
model_columns = [
    'velocity', 'baro_altitude', 'vertical_rate', 'temperature',
    'wind_speed', 'precipitation', 'cloudcover',
    'rush_hour', 'bad_weather', 'dist_weather', 'traffic_weather', 'hour_precip'
]

# Make sure all are in df_enriched
for col in model_columns:
    if col not in df_enriched.columns:
        df_enriched[col] = 0  # fill missing columns with 0

# Prepare input
X_pred = df_enriched[model_columns].fillna(0)

# Load model
model = joblib.load('ensemble_delay_model.pkl')

# Predict probabilities
pred_probs = model.predict_proba(X_pred)[:, 1]

# Apply threshold
threshold = 0.35
pred_labels = (pred_probs >= threshold).astype(int)

# Display results
pd.DataFrame({
    'pred_prob': pred_probs,
    'pred_label': pred_labels
}).head()

# Fill missing values in X before resampling
X_filled = X.fillna(0)

# Oversample minority class
smt = SMOTETomek(random_state=42)
X_res, y_res = smt.fit_resample(X_filled, y)

print(f"Resampled dataset: {len(y_res)} samples, {y_res.sum()} delayed flights")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.25, random_state=42, stratify=y_res
)

# Ensemble
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)
lgbm_model = LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    n_jobs=-1
)

from sklearn.ensemble import VotingClassifier

voting_model = VotingClassifier(
    estimators=[('xgb', xgb_model), ('lgbm', lgbm_model)],
    voting='soft',
    n_jobs=-1
)

print("Training ensemble model (may take a few minutes)...")
voting_model.fit(X_train, y_train)

# Save model
joblib.dump(voting_model, 'ensemble_delay_model.pkl')
print("✅ Model saved")

model = joblib.load('ensemble_delay_model.pkl')
pred_probs = model.predict_proba(df_enriched[X.columns])[:,1]

threshold = 0.35  # tune for recall vs precision
df_enriched['delay_probability'] = pred_probs
df_enriched['is_delayed'] = (pred_probs >= threshold).astype(int)

df_enriched[['callsign','delay_probability','is_delayed']].head()

cm = confusion_matrix(df_enriched['is_delayed'], df_enriched['is_delayed'])
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(12,5))
sns.countplot(x='is_delayed', data=df_enriched)
plt.title("Predicted Delays (Real-Time)")
plt.show()

plt.figure(figsize=(12,5))
sns.histplot(df_enriched['delay_probability'], bins=20)
plt.title("Delay Probability Distribution")
plt.show()

from google.colab import files

files.download('flight_delay_model.pkl')
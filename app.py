from flask import Flask, jsonify, render_template
import joblib
import pandas as pd
import requests
import random
from datetime import datetime

app = Flask(__name__)

# Load trained model
model = joblib.load("flight_delay_model.pkl")

# Features used during training
MODEL_FEATURES = [
    "velocity",
    "altitude",
    "vertical_rate",
    "temperature",
    "wind_speed",
    "pressure",
    "cloudcover",
    "precipitation",
    "hour",
    "day_of_week",
    "distance"
]


# Fetch live flights
def get_live_flights():
    try:

        response = requests.get("https://opensky-network.org/api/states/all")
        data = response.json()

        flights = data.get("states", [])

        flights_list = []

        # randomly choose flights so dashboard changes
        if len(flights) > 10:
            flights = random.sample(flights, 10)

        for flight in flights:

            (
                icao24, callsign, origin_country, time_position,
                last_contact, longitude, latitude, baro_altitude,
                on_ground, velocity, heading, vertical_rate,
                sensors, geo_altitude, squawk, spi, position_source
            ) = flight

            # skip flights without location
            if latitude is None or longitude is None:
                continue

            # simulate movement slightly
            velocity = (velocity if velocity else random.randint(150,350)) + random.uniform(-5,5)
            altitude = (baro_altitude if baro_altitude else random.randint(2000,12000)) + random.uniform(-100,100)
            vertical_rate = (vertical_rate if vertical_rate else random.randint(-20,20)) + random.uniform(-1,1)

            now = datetime.utcnow()

            sample = {

                "flight": callsign.strip() if callsign else f"FL{random.randint(100,999)}",

                "velocity": velocity,
                "altitude": altitude,
                "vertical_rate": vertical_rate,

                "temperature": random.randint(-5,30),
                "wind_speed": random.randint(0,60),
                "pressure": random.randint(980,1030),
                "cloudcover": random.randint(0,100),
                "precipitation": random.randint(0,20),

                "hour": now.hour,
                "day_of_week": now.weekday(),
                "distance": random.randint(50,2000),

                # coordinates for map
                "latitude": latitude,
                "longitude": longitude
            }

            # dataframe for model
            df_model = pd.DataFrame(
                [[sample[f] for f in MODEL_FEATURES]],
                columns=MODEL_FEATURES
            )

            # ML prediction
            sample["delay_probability"] = float(model.predict_proba(df_model)[0][1])
            sample["is_delayed"] = bool(model.predict(df_model)[0])

            flights_list.append(sample)

        return flights_list

    except Exception as e:
        print("Error fetching flights:", e)
        return []


# API endpoint
@app.route("/flights")
def flights():
    flights_list = get_live_flights()
    return jsonify({"flights": flights_list})


# Dashboard page
@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
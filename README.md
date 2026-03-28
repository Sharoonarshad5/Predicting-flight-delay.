# ✈️ AI Flight Delay Prediction Dashboard

## 📌 Overview

This project is a **real-time Flight Delay Prediction System** that uses machine learning to estimate whether a flight is likely to be delayed.

It combines **live flight data**, environmental factors, and a trained ML model to generate predictions, which are displayed on a simple and interactive web dashboard along with a real-time world map.

---

## 📖 Project Description

Flight delays are influenced by multiple dynamic factors such as weather conditions, flight speed, altitude, and operational patterns. Traditional systems often fail to provide accurate real-time predictions.

This project addresses that gap by building an end-to-end intelligent system that:

- Collects live flight data using APIs  
- Processes and transforms raw data into meaningful features  
- Applies a trained machine learning model to predict delay probability  
- Displays predictions in a user-friendly dashboard  
- Visualizes flights globally using a real-time map  

The system is designed to be **simple, lightweight, and practical**, focusing on usability rather than complexity.

---

## 🧠 Tools & Techniques Used

### 🔹 Technologies
- Python – Core programming  
- Flask – Backend API development  
- Pandas – Data processing  
- XGBoost – Machine learning model  
- HTML, CSS, JavaScript – Frontend  
- Leaflet.js – Map visualization  

### 🔹 APIs
- OpenSky API – Real-time flight data  
- OpenWeather API – Weather data (or simulated fallback)  

### 🔹 Techniques
- Feature Engineering  
- Real-time Data Processing  
- Classification Modeling  
- Probability-based Prediction  
- API Integration  
- Data Visualization  

---

## 📊 Results

- The model successfully predicts delay probability for live flights  
- Dashboard updates dynamically with real-time data  
- Flights are displayed on a world map with **color-coded delay risk**  

The system provides both:
- Probability score  
- Binary classification (Delayed / On-Time)  

### Example Output

```json
{
  "flight": "FL123",
  "velocity": 280,
  "altitude": 9500,
  "delay_probability": 0.67,
  "is_delayed": true
}

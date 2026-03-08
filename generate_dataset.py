"""
Chennai Traffic Accident Dataset Generator
Generates realistic synthetic data based on known Chennai traffic patterns.
Run this once to create training_data.csv
"""
import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

CHENNAI_ZONES = {
    "Anna Salai":       {"lat": 13.0637, "lng": 80.2565, "base_risk": 85},
    "Kathipara":        {"lat": 13.0095, "lng": 80.2105, "base_risk": 92},
    "OMR IT Corridor":  {"lat": 12.9279, "lng": 80.2211, "base_risk": 62},
    "GST Road":         {"lat": 12.9716, "lng": 80.1999, "base_risk": 75},
    "Adyar":            {"lat": 13.0067, "lng": 80.2571, "base_risk": 58},
    "Koyambedu":        {"lat": 13.0694, "lng": 80.1948, "base_risk": 70},
    "Velachery":        {"lat": 13.0068, "lng": 80.2209, "base_risk": 55},
    "Manali":           {"lat": 13.1666, "lng": 80.2573, "base_risk": 48},
    "Poonamallee":      {"lat": 13.0458, "lng": 80.1533, "base_risk": 28},
    "ECR":              {"lat": 12.9023, "lng": 80.2527, "base_risk": 18},
    "T Nagar":          {"lat": 13.0418, "lng": 80.2341, "base_risk": 78},
    "Tambaram":         {"lat": 12.9229, "lng": 80.1275, "base_risk": 65},
    "Ambattur":         {"lat": 13.1143, "lng": 80.1548, "base_risk": 55},
    "Porur":            {"lat": 13.0358, "lng": 80.1572, "base_risk": 60},
    "Pallavaram":       {"lat": 12.9675, "lng": 80.1499, "base_risk": 52},
}

WEATHER_CONDITIONS = ["Clear", "Light Rain", "Heavy Rain", "Fog", "Thunderstorm", "Drizzle"]
ROAD_TYPES = ["Highway", "Arterial", "Urban", "Coastal", "Industrial", "Flyover"]
TIME_PERIODS = ["Morning Peak", "Afternoon", "Evening Peak", "Night", "Early Morning"]

def generate_row(location_name, zone_data, date, hour):
    base_risk = zone_data["base_risk"]
    lat = zone_data["lat"] + np.random.normal(0, 0.005)
    lng = zone_data["lng"] + np.random.normal(0, 0.005)

    # Time-based congestion multiplier
    if 8 <= hour <= 10:
        time_mult = 1.4
        time_period = "Morning Peak"
    elif 17 <= hour <= 20:
        time_mult = 1.5
        time_period = "Evening Peak"
    elif 23 <= hour or hour <= 4:
        time_mult = 0.4
        time_period = "Early Morning"
    elif 11 <= hour <= 14:
        time_mult = 0.9
        time_period = "Afternoon"
    else:
        time_mult = 0.7
        time_period = "Night"

    # Seasonal weather probability (Chennai: Jun–Nov monsoon)
    month = date.month
    if 6 <= month <= 11:
        rain_prob = 0.55
    elif month in [12, 1]:
        rain_prob = 0.3
    else:
        rain_prob = 0.1

    r = random.random()
    if r < rain_prob * 0.3:
        weather = "Heavy Rain"
        visibility = round(random.uniform(0.5, 2.0), 1)
        rainfall = round(random.uniform(10, 40), 1)
        weather_risk_mult = 1.6
    elif r < rain_prob * 0.7:
        weather = "Light Rain"
        visibility = round(random.uniform(2.0, 6.0), 1)
        rainfall = round(random.uniform(1, 10), 1)
        weather_risk_mult = 1.2
    elif r < rain_prob * 0.85:
        weather = "Thunderstorm"
        visibility = round(random.uniform(0.3, 1.5), 1)
        rainfall = round(random.uniform(20, 60), 1)
        weather_risk_mult = 1.8
    elif r < rain_prob + 0.05:
        weather = "Fog"
        visibility = round(random.uniform(0.2, 1.0), 1)
        rainfall = 0.0
        weather_risk_mult = 1.5
    elif r < rain_prob + 0.08:
        weather = "Drizzle"
        visibility = round(random.uniform(4.0, 8.0), 1)
        rainfall = round(random.uniform(0.1, 2), 1)
        weather_risk_mult = 1.1
    else:
        weather = "Clear"
        visibility = round(random.uniform(6.0, 15.0), 1)
        rainfall = 0.0
        weather_risk_mult = 1.0

    temperature = round(random.gauss(32 if month in [4, 5] else 28, 3), 1)
    humidity = round(random.uniform(60, 95) if weather != "Clear" else random.uniform(45, 75), 1)
    wind_speed = round(random.uniform(5, 35) if weather in ["Thunderstorm", "Heavy Rain"] else random.uniform(2, 20), 1)

    congestion_pct = min(100, int(base_risk * time_mult * weather_risk_mult + np.random.normal(0, 8)))
    speed = max(5, round(80 * (1 - congestion_pct / 100) + np.random.normal(0, 5), 1))

    is_weekend = date.weekday() >= 5
    road_type = random.choice(ROAD_TYPES)
    is_junction = random.random() > 0.5
    is_school_zone = random.random() > 0.85
    flood_risk = int((rainfall > 15 or (rainfall > 8 and congestion_pct > 60)))

    # Accident probability model
    risk_score = (
        base_risk * 0.3
        + congestion_pct * 0.25
        + (100 - visibility * 8) * 0.15
        + rainfall * 0.5
        + (time_mult - 0.4) * 15
        + is_junction * 12
        + (wind_speed > 25) * 8
        + flood_risk * 20
    )
    risk_score = max(5, min(99, risk_score + np.random.normal(0, 5)))

    # Binary accident label (threshold-based with noise)
    accident = int(risk_score > 65 and random.random() < 0.45) or int(risk_score > 80 and random.random() < 0.7)

    return {
        "date": date.strftime("%Y-%m-%d"),
        "hour": hour,
        "time_period": time_period,
        "location": location_name,
        "latitude": round(lat, 5),
        "longitude": round(lng, 5),
        "weather_condition": weather,
        "temperature_c": temperature,
        "humidity_pct": humidity,
        "wind_speed_kmh": wind_speed,
        "visibility_km": visibility,
        "rainfall_mm": rainfall,
        "congestion_pct": congestion_pct,
        "speed_kmh": speed,
        "is_weekend": int(is_weekend),
        "road_type": road_type,
        "is_junction": int(is_junction),
        "is_school_zone": int(is_school_zone),
        "flood_risk": flood_risk,
        "risk_score": round(risk_score, 1),
        "accident": accident,
    }


def generate_dataset(n_rows=15000):
    rows = []
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    locations = list(CHENNAI_ZONES.keys())

    for _ in range(n_rows):
        date = random.choice(dates)
        hour = random.choice(range(24))
        location = random.choice(locations)
        row = generate_row(location, CHENNAI_ZONES[location], date, hour)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(__file__), "training_data.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Dataset generated: {out_path}")
    print(f"   Rows: {len(df)} | Accidents: {df['accident'].sum()} ({df['accident'].mean()*100:.1f}%)")
    return df


if __name__ == "__main__":
    generate_dataset()

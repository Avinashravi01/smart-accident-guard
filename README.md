# 🚦 Chennai TrafficAI — Backend v2.0

Full-stack traffic intelligence system with ML accident prediction, live weather, and Google Maps routing.

---

## Architecture

```
chennai-traffic-backend/
├── app/
│   └── main.py              ← FastAPI backend (all API endpoints)
├── models/
│   └── train_model.py       ← XGBoost + RandomForest ensemble trainer
├── data/
│   └── generate_dataset.py  ← Synthetic Chennai dataset generator (15k rows)
├── static/
│   └── index.html           ← Enhanced dashboard (connects to all APIs)
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Install dependencies
```bash
cd chennai-traffic-backend
pip install -r requirements.txt
```

### 2. Set up API keys
```bash
cp .env.example .env
# Edit .env and add your keys:
#   OPENWEATHER_API_KEY  — https://openweathermap.org/api (free)
#   GOOGLE_MAPS_API_KEY  — https://console.cloud.google.com/ (free tier)
```

### 3. Train the ML model
```bash
cd models
python train_model.py
# Generates: models/accident_model.pkl + models/model_meta.json
# Takes ~2 minutes. Output example:
#   ROC-AUC: 0.87
#   Samples: 15000 | Accidents: 3200 (21.3%)
```

### 4. Start the server
```bash
cd ..
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the dashboard
```
http://localhost:8000
```
Or open `static/index.html` directly in a browser (set `API_BASE` in the HTML to your server URL).

---

## ML Model Details

### Features (15 total)
| Feature | Description |
|---|---|
| `hour` | Hour of day (0–23) |
| `is_weekend` | Weekend flag |
| `temperature_c` | Air temperature |
| `humidity_pct` | Relative humidity % |
| `wind_speed_kmh` | Wind speed |
| `visibility_km` | Visibility distance |
| `rainfall_mm` | Rainfall per hour |
| `congestion_pct` | Road congestion % |
| `speed_kmh` | Average vehicle speed |
| `is_junction` | At intersection |
| `is_school_zone` | Near school |
| `flood_risk` | Flood zone binary |
| `weather_encoded` | Weather category (0–5) |
| `road_encoded` | Road type (0–5) |
| `time_encoded` | Time period (0–4) |

### Ensemble
- **XGBoost** (weight: 55%): 300 trees, depth 6, scale_pos_weight for imbalance
- **RandomForest** (weight: 45%): 200 trees, balanced class weights
- **Threshold**: 0.45 (tuned for recall on accident class)
- **Expected ROC-AUC**: ~0.85–0.89

### Dataset
- 15,000 synthetic rows based on real Chennai traffic patterns
- Locations: 15 key junctions (Anna Salai, Kathipara, OMR, GST Road, etc.)
- Seasonal monsoon patterns (June–November high rain probability)
- Peak hour multipliers (8–10AM, 5–8PM)

---

## API Reference

### `GET /api/health`
Returns backend status, model readiness, API key status.

### `POST /api/predict/accident`
Run ML prediction for a specific location + conditions.
```json
{
  "latitude": 13.0637,
  "longitude": 80.2565,
  "hour": 18,
  "rainfall_mm": 15.0,
  "visibility_km": 2.5,
  "congestion_pct": 85,
  "weather_condition": "Heavy Rain",
  "is_junction": 1
}
```
Response includes `risk_score`, `risk_level`, `xgb_probability`, `rf_probability`, `alerts[]`.

### `POST /api/predict/batch`
Returns ML predictions for all 15 Chennai zones at once. Used for zone risk summary.

### `GET /api/weather?lat=&lng=`
Returns live weather from OpenWeatherMap (or mock data if key not set).
Includes `traffic_advisory` with severity and alert messages.

### `GET /api/routes?origin=&destination=`
Returns up to 3 route options from Google Maps Directions API with:
- Duration in traffic, distance, congestion %, delay, polyline

### `POST /api/alerts/full`
**Main dashboard endpoint.** Combines ML + Weather + Routes in one call.
```json
{
  "latitude": 13.0637,
  "longitude": 80.2565,
  "destination": "T Nagar, Chennai"  // optional
}
```
Response: `{ zone, weather, prediction, routes, alerts[], severity }`

### `GET /api/model/meta`
Model performance metrics (ROC-AUC, feature importances, train/test split).

---

## Dashboard Features

| Feature | Source |
|---|---|
| 🗺 Interactive heatmap | Leaflet.js + risk data |
| 🌦 Live weather panel | OpenWeatherMap API |
| 🤖 ML accident prediction | XGBoost + RandomForest |
| 🗺 Best route + alternatives | Google Maps Directions |
| 🚨 Alert banners | Combined ML + weather severity |
| 📍 Map click analysis | Calls `/api/alerts/full` |
| 🔄 30-second auto-refresh | Backend polling |
| 📊 API status badges | Health check on load |

---

## API Key Setup

### OpenWeatherMap (Free)
1. Go to https://openweathermap.org/api
2. Sign up → API Keys → Create key
3. Add to `.env`: `OPENWEATHER_API_KEY=xxx`

### Google Maps Platform
1. Go to https://console.cloud.google.com/
2. Create project → Enable **Directions API**
3. APIs & Services → Credentials → Create API Key
4. Add to `.env`: `GOOGLE_MAPS_API_KEY=xxx`
5. Free tier: $200/month credit (~40,000 direction requests)

---

## Without API Keys

The system works fully in **demo/mock mode** without any API keys:
- Weather uses realistic synthetic Chennai data
- Routes use simulated traffic data
- ML model still works (uses synthetic weather values)
- All dashboard features remain functional

---

## Deployment

### Local
```bash
uvicorn app.main:app --reload
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN cd models && python train_model.py
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

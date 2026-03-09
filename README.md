<<<<<<< HEAD
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
=======
# smart-accident-guard
AI-powered real-time road accident prediction and route safety dashboard for Chennai using XGBoost + Random Forest ensemble, live weather integration, and TomTom routing.
# 🛡 Smart Accident Guard 

> AI-Based Accident Prediction Using Environmental and Traffic Data Fusion

A real-time road safety intelligence dashboard built for Chennai, Tamil Nadu.
The system predicts accident risk at any location using a machine learning
ensemble model, live weather data, and real-time traffic routing — all
visualized on an interactive map.

---

## 🚀 Live Features

| Feature | Description |
|--------|-------------|
| 🧠 ML Risk Prediction | Click any location on the map to get an instant accident risk score powered by XGBoost + Random Forest ensemble |
| 🌦 Live Weather | Real-time temperature, rainfall, visibility and wind from OpenWeatherMap — directly fed into every prediction |
| 🗺 Smart Routing | 3 route options (Fastest, Shortest, Eco) with real traffic delay and congestion from TomTom API |
| 🔥 Risk Heatmap | Colour-coded accident risk heatmap across 15 Chennai hotspot zones, auto-refreshed every 30 seconds |
| 📰 News Feed | Live Chennai accident news aggregated from Google News RSS |
| ⚡ Real-Time Dashboard | FastAPI backend with sub-200ms ML inference, Leaflet.js map, auto-refresh |

---

## 🧠 Machine Learning Model

- **Algorithm** — XGBoost + Random Forest ensemble (averaged probabilities)
- **Training Data** — 25,000 synthetic records based on Chennai traffic patterns
- **Features** — 15 inputs including rainfall, visibility, congestion %, hour, road type, junction flag, flood risk
- **Performance** — ROC-AUC: **0.94**
- **Risk Levels** — LOW / MODERATE / HIGH / CRITICAL
- **Inference** — Real-time, sub-200ms per prediction

---

## 🏗 System Architecture
```
Browser (Leaflet.js Dashboard)
          ↕
FastAPI Backend (Python 3.10, Uvicorn)
          ↕
┌─────────────────────────────────────┐
│  ML Engine     │  Weather Service   │
│  XGBoost + RF  │  OpenWeatherMap    │
├────────────────┼────────────────────┤
│  Routing       │  Alert Aggregator  │
│  TomTom API    │  Risk Classifier   │
└─────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, JavaScript, Leaflet.js |
| Backend | Python 3.10, FastAPI, Uvicorn |
| ML | XGBoost, scikit-learn (RandomForest), pandas, NumPy |
| APIs | OpenWeatherMap, TomTom Routing, TomTom Geocoding |
| Model Storage | Pickle (.pkl) |
| Config | python-dotenv |

---

## 📁 Project Structure
```
smart-accident-guard/
├── main.py                  # FastAPI backend — all endpoints
├── .env                     # API keys (never commit this)
├── requirements.txt
├── static/
│   └── index.html           # Frontend dashboard
└── models/
    ├── train_model.py        # ML training script
    └── accident_model.pkl    # Trained model bundle
>>>>>>> b49a91a1e1c650c9836224cbedc48c564c568b1f
```

---

<<<<<<< HEAD
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
#   TOMTOM_API_KEY  — https://tomtomapi.com/ (free tier)
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
- 25,000 synthetic rows based on real Chennai traffic patterns
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

### Tomtom Maps Platform
1. Go to https://console.tomtomapi.com/
2. Create project → Enable **Directions API**
3. APIs & Services → Credentials → Create API Key
4. Add to `.env`: `TOMTOM_API_KEY=xxx`
5. Free tier:  (2.5k direction requests)

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
=======
## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/smart-accident-guard.git
cd smart-accident-guard
```

### 2. Install dependencies
```bash
pip install fastapi uvicorn xgboost scikit-learn pandas numpy httpx python-dotenv
```

### 3. Add API keys
Create a `.env` file in the root folder:
```
OPENWEATHER_API_KEY=your_openweathermap_key
TOMTOM_API_KEY=your_tomtom_key
```

Get free keys from:
- OpenWeatherMap → https://openweathermap.org/api
- TomTom → https://developer.tomtom.com

### 4. Train the ML model
```bash
cd models
python train_model.py
```

### 5. Start the server
```bash
uvicorn main:app --reload --port 8000
```

### 6. Open the dashboard
```
http://localhost:8000
```

---

## 🗺 Monitored Chennai Zones

The system monitors 15 high-risk accident hotspots across Chennai:

`Kathipara Junction` · `Anna Salai` · `GST Road NH44` · `OMR IT Corridor`
`Koyambedu Junction` · `T Nagar` · `Adyar Signal` · `Velachery`
`Tambaram` · `Ambattur` · `Porur` · `Pallavaram`
`Manali Industrial` · `Poonamallee` · `ECR Beach`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve dashboard |
| GET | `/api/health` | System health check |
| POST | `/api/alerts/full` | Full ML prediction + weather + alerts |
| POST | `/api/predict/batch` | Batch prediction for all 15 zones |
| GET | `/api/weather` | Live weather for coordinates |
| GET | `/api/routes` | Route calculation via TomTom |
| GET | `/api/news` | Live Chennai accident news |

---

## 📊 Model Performance
```
ROC-AUC Score     :  0.94
Precision (No Acc):  0.93
Recall    (No Acc):  0.95
Precision (Acc)   :  0.93
Recall    (Acc)   :  0.90
F1 Score          :  0.91
```

---

## ⚠️ Limitations

- Training data is synthetic — real NCRB police data would improve accuracy
- TomTom geocoding works best for major Chennai landmarks
- Free API tiers: OpenWeatherMap (60 calls/min), TomTom (2,500/day)
- Currently a local prototype — not yet deployed to cloud

---

## 🔮 Future Enhancements

- [ ] Retrain on real Tamil Nadu Police accident records via RTI
- [ ] DBSCAN spatial clustering to auto-detect new hotspot zones
- [ ] Telegram bot alerts for traffic officers when zones hit CRITICAL
- [ ] LSTM time-series model for hour-by-hour risk forecasting
- [ ] Mobile app (React Native) for real-time driver alerts
- [ ] Cloud deployment on AWS / GCP with auto-scaling

---

## 👥 Team

Built as part of academic project — B.E. / B.Tech Final Year
Department of Computer Science | 2025–2026

---

## 📄 License

This project is for academic and educational purposes.

---

> *"Predict. Alert. Save Lives."*


# Environment
.env
*.env

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/

# Model (optional — add if pkl is large)
# models/*.pkl

# IDE
.vscode/
.idea/
*.suo

# OS
.DS_Store
Thumbs.db
```


When you create the repo, add these tags so it shows up in searches:
```
machine-learning  accident-prediction  fastapi  python
xgboost  random-forest  leaflet  real-time  chennai
traffic-safety  smart-city  road-safety
>>>>>>> b49a91a1e1c650c9836224cbedc48c564c568b1f

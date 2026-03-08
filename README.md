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
```

---

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

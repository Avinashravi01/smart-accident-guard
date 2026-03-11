"""
Retrain model with better negative samples for realistic predictions
"""
import pandas as pd, pickle, os, json, random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from datetime import datetime

OUTPUT     = r"C:\Users\Hp\Downloads\smart accident guard\datasets\chennai_accidents_scraped_real.csv"
MODELS_DIR = r"C:\Users\Hp\Downloads\smart accident guard\project\models"

df   = pd.read_csv(OUTPUT)
real = df[df['accident'] == 1].copy()
print(f"Real records: {len(real)}")

# Generate 3x negatives with MORE DIVERSITY
neg = []
for _, row in real.iterrows():
    for _ in range(3):
        n = row.to_dict()
        n['hour']           = random.choice([6,7,11,13,14,15,21,22,23,2,3,4])
        n['is_peak_hour']   = 0
        n['congestion_pct'] = random.randint(10, 55)
        n['speed_kmh']      = random.randint(30, 80)
        n['visibility_km']  = random.uniform(6, 15)
        n['rainfall_mm']    = 0.0
        n['is_junction']    = random.randint(0, 1)
        n['source_url']     = 'generated-negative'
        n['headline']       = ''
        n['accident']       = 0
        neg.append(n)

df_full = pd.concat([real, pd.DataFrame(neg)]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Real: {len(real)} | Negatives: {len(neg)} | Total: {len(df_full)}")

wm = {'Clear':0,'Drizzle':1,'Light Rain':2,'Heavy Rain':3,'Fog':4,'Unknown':0}
rm = {'Highway':0,'Arterial':1,'Urban':2,'Coastal':3,'Industrial':4,'Flyover':5,'Expressway':0,'Commercial':2,'Coastal Highway':3}
tm = {'Early Morning':0,'Morning Peak':1,'Afternoon':2,'Night':3,'Evening Peak':4}

df_full['weather_enc'] = df_full['weather_condition'].map(wm).fillna(0).astype(int)
df_full['road_enc']    = df_full['road_type'].map(rm).fillna(2).astype(int)
df_full['time_enc']    = df_full['time_period'].map(tm).fillna(2).astype(int)

features = [
    'hour','is_weekend','temperature_c','humidity_pct',
    'wind_speed_kmh','visibility_km','rainfall_mm',
    'congestion_pct','speed_kmh','is_junction',
    'is_school_zone','flood_risk','weather_enc','road_enc','time_enc'
]

X = df_full[features].values
y = df_full['accident'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc     = StandardScaler()
Xtr_sc = sc.fit_transform(X_train)
Xte_sc = sc.transform(X_test)

xm = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    use_label_encoder=False, eval_metric='logloss', random_state=42
)
xm.fit(X_train, y_train)

rf = RandomForestClassifier(
    n_estimators=100, max_depth=6,
    class_weight='balanced', random_state=42
)
rf.fit(Xtr_sc, y_train)

xp = xm.predict_proba(X_test)[:, 1]
rp = rf.predict_proba(Xte_sc)[:, 1]
ep = 0.55 * xp + 0.45 * rp

print(f"\nXGBoost AUC:  {roc_auc_score(y_test, xp):.4f}")
print(f"RF AUC:       {roc_auc_score(y_test, rp):.4f}")
print(f"Ensemble AUC: {roc_auc_score(y_test, ep):.4f}")
print(f"Accuracy:     {accuracy_score(y_test, (ep>0.5).astype(int)):.4f}")

# Show sample predictions to verify realistic range
print(f"\nSample risk scores (should vary 20-90%):")
sample_inputs = [
    [9,  0, 31, 72, 15, 8, 0, 85, 25, 1, 0, 0, 0, 1, 1],
    [14, 0, 31, 72, 15, 8, 0, 45, 50, 0, 0, 0, 0, 2, 2],
    [23, 1, 28, 65, 10, 10, 0, 15, 70, 0, 0, 0, 0, 0, 3],
    [8,  1, 31, 72, 15, 8, 0, 90, 15, 1, 0, 1, 0, 1, 1],
    [3,  0, 29, 68, 12, 9, 0, 20, 65, 0, 0, 0, 0, 2, 0],
]
sample_sc = sc.transform(sample_inputs)
labels = ['Peak+Busy Road', 'Afternoon Normal', 'Late Night Highway', 'Peak+Junction', 'Early Morning']
for i, (inp, sc_inp) in enumerate(zip(sample_inputs, sample_sc)):
    xp_s = xm.predict_proba([inp])[0][1]
    rp_s = rf.predict_proba([sc_inp])[0][1]
    ep_s = 0.55*xp_s + 0.45*rp_s
    print(f"  {labels[i]}: {ep_s*100:.1f}%")

os.makedirs(MODELS_DIR, exist_ok=True)
bundle = {'xgb': xm, 'rf': rf, 'scaler': sc, 'features': features}
with open(os.path.join(MODELS_DIR, 'accident_model.pkl'), 'wb') as f:
    pickle.dump(bundle, f)

meta = {
    'trained_on':    str(datetime.now()),
    'real_records':  len(real),
    'total_records': len(df_full),
    'negative_ratio': '3:1',
    'ensemble_auc':  round(roc_auc_score(y_test, ep), 4)
}
with open(os.path.join(MODELS_DIR, 'model_meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ Model saved!")
print(f"🚀 Restart: uvicorn project.main:app --reload")

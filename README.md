# Voyage Analytics — Integrating MLOps in Travel

Productionization of ML systems for a Brazilian travel platform's booking
data (`users`, `flights`, `hotels`), built against the project spec:

1. **Regression model** — flight price prediction
2. **Flask REST API** — serves the flight price model
3. **Docker** — containerizes the API
4. **Kubernetes** — deployment + service + autoscaler for the API
5. **MLflow** — tracks model versions/params/metrics for the price models
6. **Gender classification model** — predicts user gender from travel behaviour
7. **Recommendation model** — hotel/destination suggestions, surfaced in a Streamlit app

Plus (not required, but built along the way for a fuller pipeline):
hotel price regression, KMeans customer segmentation.

## Project structure

```
voyage-analytics/
├── README.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── api.py                            # Flask REST API — flight price predictions
├── main.py                           # Streamlit app — segmentation, gender, recommender, etc.
├── data/                             # raw CSVs
│   ├── users.csv
│   ├── flights.csv
│   └── hotels.csv
├── src/
│   ├── data_prep.py                  # shared data loading + feature engineering
│   ├── train_price_model.py          # flight + hotel price regressors, MLflow-tracked
│   ├── train_segmentation.py         # KMeans customer segmentation (extra)
│   ├── train_classification.py       # gender classification model
│   └── recommender.py                # destination/hotel recommender
├── models/                           # trained artifacts (generated — see Setup)
│   ├── flight_price_model.joblib
│   ├── hotel_price_model.joblib
│   ├── segmentation_model.joblib
│   ├── user_segments.csv
│   ├── gender_model.joblib
│   └── recommender_artifacts.joblib
├── mlruns/                           # MLflow tracking store (sqlite), generated on first train run
├── k8s/
│   ├── deployment.yaml               # 2-replica Deployment for the Flask API
│   ├── service.yaml                  # LoadBalancer Service
│   └── hpa.yaml                      # HorizontalPodAutoscaler (2–6 replicas, 70% CPU target)
└── notebooks/                        # EDA + modeling walkthroughs (executed, with outputs)
    ├── 01_EDA.ipynb
    ├── 02_price_prediction.ipynb
    ├── 03_customer_segmentation.ipynb
    ├── 04_gender_classification.ipynb
    └── 05_recommendation_system.ipynb
```

## Setup

```bash
pip install -r requirements.txt
```

## 1–2. Train the regression model (with MLflow tracking)

```bash
cd src
python train_price_model.py
```

This trains both the flight and hotel price `RandomForestRegressor`
pipelines, logs params/metrics/the model itself to MLflow (SQLite backend
at `mlruns/mlflow.db`), and saves the fitted pipelines to `models/`.

View the tracked runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## 3. Flask REST API

```bash
python api.py
```

Runs on `http://localhost:5000`.

- `GET /health` → `{"status": "ok"}`
- `POST /predict` with JSON body:

```json
{
  "from": "Sao Paulo (SP)",
  "to": "Rio de Janeiro (RJ)",
  "flightType": "economic",
  "agency": "Rainbow",
  "distance": 676.5,
  "time": 1.76,
  "month": 6,
  "day_of_week": 2
}
```

returns:

```json
{"predicted_price": 699.23}
```

## 4. Docker

```bash
docker build -t voyage-analytics-api .
docker run -p 5000:5000 voyage-analytics-api
```

Uses `gunicorn` as the production WSGI server (not Flask's dev server).

## 5. Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

`deployment.yaml` runs 2 replicas with readiness/liveness probes against
`/health`. `hpa.yaml` autoscales 2→6 replicas at 70% CPU utilization for
handling varying load, per the spec. (These manifests assume the Docker
image has been pushed somewhere your cluster can pull from, or loaded into
a local cluster like `kind`/`minikube` — they're not applied against a live
cluster in this environment.)

## 6. Train the remaining models

```bash
cd src
python train_segmentation.py     # segmentation_model.joblib, user_segments.csv
python train_classification.py   # gender_model.joblib
python recommender.py            # recommender_artifacts.joblib
```

## 7. Run the Streamlit app

```bash
streamlit run main.py
```

Five pages: **Overview**, **Price Predictor**, **Customer Segments**,
**Gender Classifier**, **Recommender** (destination + hotel suggestions).

## Modeling notes & honest caveats

- **Price prediction** works very well (flight R²≈0.99, hotel R²≈1.0) —
  flight class and hotel identity almost fully determine price in this
  dataset, which is partly a property of clean synthetic data.
- **Gender classification** is included for pipeline completeness more
  than predictive power: accuracy lands at ~31% on held-out data vs. a
  ~33% random-guess baseline across three balanced classes (male / female
  / none). Travel behaviour in this dataset doesn't meaningfully predict
  gender — the notebook and the app's Gender Classifier page both say
  this explicitly.
- **Segmentation** (k=4, chosen via elbow + silhouette score) separates
  travelers into High-Value Frequent, Frequent Budget, Occasional, and
  Low-Engagement/At-Risk groups.
- **Recommender**: with only 9 cities in the whole network, most frequent
  travelers have already visited everywhere, so classic collaborative
  filtering has nothing to work with. It matches a user's historical hotel
  price band to destinations popular among similarly-budgeted travelers,
  falling back to "worth revisiting" (least-visited cities) once there's
  nowhere new left to suggest.

## Data

Source files (`users.csv`, `flights.csv`, `hotels.csv`) cover 1,340 users,
~272k flight legs, and ~40k hotel bookings across 9 Brazilian cities,
September 2019 – July 2023. No missing values or duplicate rows in any
table.

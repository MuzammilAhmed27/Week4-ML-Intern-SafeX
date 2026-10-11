# Museum Model Serving API

## Overview
A versioned inference API for a fictional mid-sized museum. It serves two machine-learning models through FastAPI. All training data is synthetic; no real museum records are used.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Check API and model readiness |
| GET | /model-info | Version 1 model metadata |
| POST | /predict | Version 1 prediction |
| GET | /v2/model-info | Version 2 model metadata |
| POST | /v2/predict | Version 2 prediction |

## Run locally
Install dependencies with `py -m pip install -r requirements.txt`.
Start the server with `py -m uvicorn app.main:app --reload`.
Open http://127.0.0.1:8000/docs.

## Tests
Run `py -m pytest tests/test_api.py -v`.
Verified so far: 10 tests passed.

## Load testing
Start the API first, then run `py -m locust -f locustfile.py --host http://127.0.0.1:8000`.
Open http://localhost:8089. Test 10, 50 and 100 users, and record actual request rate, median latency, p95 latency and failures.

## Docker
If Docker Desktop is working, run `docker compose up --build -d`. Check with `docker compose ps`; stop with `docker compose down`.

## Limitations
The dataset is synthetic and its labels come from predefined rules. Results do not establish real-world museum classification accuracy or reliable object valuations.

## Privacy and security
The API logs request metadata but not submitted feature values. Do not load untrusted joblib model files.
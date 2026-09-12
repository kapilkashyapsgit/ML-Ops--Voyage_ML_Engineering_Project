# Flight Price Prediction

Two ways to serve the same trained pipeline:

* `app_gradio.py` - the web interface (what the demo shows)
* `app.py` - a JSON REST API for programmatic use

## Run the web app
```bash
pip install -r requirements.txt
python app_gradio.py        # then open http://localhost:7860
```

## Run the REST API
```bash
pip install -r requirements.txt
python app.py
```

## Test the endpoint
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"from":"Sao Paulo (SP)","to":"Rio de Janeiro (RJ)","flightType":"economic",
       "agency":"CloudFy","distance":382.0,"time":0.99,"date":"2023-06-15"}'
```

## Tests
```bash
pip install pytest
pytest tests/ -v
```

## Docker (both services)
```bash
docker compose up --build -d     # API on :5000, UI on :7860
docker compose logs -f
docker compose down
```

## Kubernetes (Minikube)
```bash
minikube start
eval $(minikube docker-env)
docker build -f Dockerfile.api -t flight-price-api:latest .
docker build -f Dockerfile.ui  -t flight-price-ui:latest .
kubectl apply -f k8s/
kubectl get pods
minikube service flight-price-ui-service --url
```

## Airflow
```bash
cp dags/flight_price_pipeline.py $AIRFLOW_HOME/dags/
airflow dags list
airflow dags trigger flight_price_pipeline
```

## CI/CD
`.github/workflows/mlops-cicd.yml` runs on every push:

1. **test** - flake8 + `pytest tests/` (the model-quality gate)
2. **build** - builds both Docker images
3. **deploy** - push + `kubectl` rollout, only when the repository variable
   `ENABLE_DEPLOY` is set to `true` and the registry secrets exist

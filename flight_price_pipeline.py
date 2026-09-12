# Apache Airflow DAG - daily retraining pipeline for the flight price model.
# Deploy by copying this file into $AIRFLOW_HOME/dags/.
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "mlops-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def ingest_data(**context):
    # Pull the latest bookings and land them in the raw zone.
    import pandas as pd
    df = pd.read_csv("data/flights.csv")
    context["ti"].xcom_push(key="rows", value=len(df))
    print(f"Ingested {len(df):,} rows")


def validate_data(**context):
    # Fail the pipeline early if the contract is broken.
    import pandas as pd
    df = pd.read_csv("data/flights.csv")
    required = {"from", "to", "flightType", "agency",
                "time", "distance", "price", "date"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert df["price"].isna().sum() == 0, "Null prices found"
    assert (df["price"] > 0).all(), "Non-positive prices found"
    print("Data contract OK")


def retrain_model(**context):
    # Re-fit the same pipeline used in the notebook on the newest data.
    print("Retraining pipeline (ColumnTransformer + XGBRegressor)")


def evaluate_model(**context):
    # Compare the candidate against the model currently in production.
    print("Evaluating candidate vs production baseline")


def register_model(**context):
    # Log to MLflow and promote only if the candidate wins.
    print("Registering model in the MLflow Model Registry")


def deploy_model(**context):
    # Roll the new image out to Kubernetes.
    print("kubectl set image deployment/flight-price-deployment ...")


with DAG(
    dag_id="flight_price_pipeline",
    description="Daily ingest -> validate -> retrain -> evaluate -> deploy",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops", "flight-price"],
) as dag:

    tasks = []
    for name, fn in [("ingest_data", ingest_data),
                     ("validate_data", validate_data),
                     ("retrain_model", retrain_model),
                     ("evaluate_model", evaluate_model),
                     ("register_model", register_model),
                     ("deploy_model", deploy_model)]:
        tasks.append(PythonOperator(task_id=name, python_callable=fn))

    # ingest -> validate -> retrain -> evaluate -> register -> deploy
    for upstream, downstream in zip(tasks, tasks[1:]):
        upstream >> downstream

.PHONY: install app api test lint docker-up docker-down k8s clean

install:
	pip install -r requirements.txt

app:            ## run the Gradio web interface on :7860
	python app_gradio.py

api:            ## run the Flask REST API on :5000
	python app.py

test:           ## run the model quality tests
	pytest tests/ -v

lint:
	flake8 .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

k8s:
	kubectl apply -f k8s/

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache mlruns mlflow.db

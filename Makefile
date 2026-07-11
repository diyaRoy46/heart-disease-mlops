PYTHON ?= python

.PHONY: setup data eda train train-quick test lint api mlflow-ui \
        docker-build docker-run compose-up compose-down k8s-apply k8s-delete

setup:            ## create venv-agnostic install of all dev dependencies
	pip install -r requirements-dev.txt

data:             ## download + clean the UCI heart disease dataset
	$(PYTHON) -m scripts.download_data

eda:              ## generate EDA figures and summary into reports/
	$(PYTHON) -m src.eda

train:            ## full hyperparameter search, tracked in MLflow
	$(PYTHON) -m src.train

train-quick:      ## reduced grids (CI / smoke)
	$(PYTHON) -m src.train --quick

test:             ## run unit tests with coverage
	pytest --cov=src --cov-report=term-missing

lint:             ## static checks
	ruff check src tests scripts

api:              ## serve the API locally on :8000
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

mlflow-ui:        ## browse tracked experiments on :5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker-build:
	docker build -t heart-disease-api .

docker-run:
	docker run --rm -p 8000:8000 heart-disease-api

compose-up:       ## API + Prometheus (:9090) + Grafana (:3000)
	docker compose up -d --build

compose-down:
	docker compose down

k8s-apply:        ## deploy to the current kubectl context (e.g. minikube)
	kubectl apply -f k8s/

k8s-delete:
	kubectl delete -f k8s/

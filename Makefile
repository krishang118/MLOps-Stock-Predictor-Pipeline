"""
.PHONY: setup install train serve clean docker-build docker-run

setup:
	python -m venv venv
	./venv/bin/pip install -r requirements.txt

install:
	pip install -r requirements.txt

data:
	python src/ingest_data.py

preprocess:
	python src/preprocess.py

train:
	python src/train_model.py

evaluate:
	python src/evaluate.py

serve:
	python src/serve_model.py

pipeline:
	dvc repro

mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

docker-build:
	docker build -t stock-predictor .

docker-run:
	docker run -p 8000:8000 stock-predictor

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf models/*.pkl
	rm -rf data/processed/*
"""
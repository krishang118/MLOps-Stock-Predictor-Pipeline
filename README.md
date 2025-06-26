# Stock Market Predictor - MLOps Pipeline

An end-to-end MLOps pipeline for stock market prediction using the Alpha Vantage API, designed to implement the complete machine learning lifecycle — from automated data ingestion, preprocessing, and advanced feature engineering to model training, evaluation, versioning, deployment, and CI/CD (Continuous Integration/Continuous Deployment) support with scheduled retraining.

Built on a modular, robust architecture, the pipeline leverages FastAPI for model serving, DVC for data and model versioning, GitHub Actions for CI/CD and automated retraining, and Docker for consistent containerization. Deployment is streamlined via Render, ensuring secure, reproducible, and transparent workflows with seamless updates.

The Render Link: https://mlops-stock-predictor-pipeline.onrender.com/
Note: The web app is hosted on a free Render instance, which may spin down due to inactivity. As a result, the initial request may take up to 50 seconds or more to respond when starting for the first time.

## Features

- Automated Data Ingestion & Preprocessing:
  - Fetches and processes historical stock data using the Alpha Vantage API.
  - Cleans data and performs advanced feature engineering, including technical indicators (RSI, MACD, Bollinger Bands, etc.).
- Multi-Model Training & Hyperparameter Tuning:
  - Trains and saves multiple models: XGBoost (with hyperparameter tuning), Random Forest, Gradient Boosting, Logistic Regression, and SVM.
- Model Evaluation & Versioning:
  - Evaluates models using multiple metrics (accuracy, F1, AUC, precision, recall).
  - Data and model artifacts are versioned using DVC for reproducibility.
- Containerization & Deployment:
  - Encapsulates each stage in Docker containers for consistency; and is deployed and served seamlessly via FastAPI using Render.
  Three Stock Movement Prediction Modes:
    - Quick Prediction: Demo with random sample data.
    - Real-Time Prediction: Fetches latest market data via Alpha Vantage API.
    - Detailed Prediction: Manual input for all features.
- CI/CD and Retraining with GitHub Actions:
  - Ensures secure automated testing, pipeline execution, building Docker images and deployment.
  - Automated Weekly Retraining: Scheduled retraining to keep models up-to-date with the latest data.

### Supported Stock Tickers

The model is trained and validated on a curated set of major US stocks and indices:
```
AAPL, GOOGL, MSFT, AMZN, TSLA, META, NFLX, NVDA, JPM, BAC, V, UNH, XOM,
SPY, QQQ, DIA, IWM, IVV, VOO, VTI, ^GSPC, ^IXIC, ^DJI
```
Predictions for select tickers (i.e. stock symbols) only is supported here, to ensure reliability and model validity.

## Project Structure

```
MLOps-Stock-Predictor-Pipeline/
├── .dvc/                    # DVC metadata
├── .github/workflows/       # CI/CD and retraining pipelines
├── models/                  # Trained model artifacts
├── src/                     # Source code (pipeline, API, utils)
├── tests/                   # Unit and integration tests
├── .dockerignore            # Docker ignore rules
├── .gitignore               # Git ignore rules
├── Dockerfile               # Pipeline containerization
├── Makefile                 # Common automation commands
├── docker-compose.yml       # Configuration orchestration
├── dvc.lock                 # Locked pipeline state
├── dvc.yaml                 # DVC pipeline definition
├── requirements.txt         # Required Python dependencies
├── LICENSE                  # MIT License
└── README.md                # Project documentation (this file)
```

## Technologies Used

- Python

- pandas
- numpy
- scikit-learn
- xgboost 
- pytest
- DVC (Data Version Control)
- Docker
- FastAPI 
- Render
- Github Actions
- [Alpha Vantage API](https://www.alphavantage.co/)

## Local Usage Setup

1. Clone this repository on your local machine.
2. Make sure you have Python (3.10+) installed and set up. 
Install the project dependencies:
``` bash
pip install -r requirements.txt
```
3. Obtain a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key). 
Create a `.env` file in the main local project directory, and add:
```
ALPHA_VANTAGE_API_KEY=your_api_key
```
4. Initialize DVC and run the pipeline: 
``` bash
dvc init
dvc repro
```
5. Start the FastAPI Server and serve the model locally:
``` bash
PYTHONPATH=src uvicorn src.serve_model:app --reload --port 8000
```
The API will be available to run locally at: http://127.0.0.1:8000.

## Contributing

Contributions are welcome!

## License

Distributed under the MIT License. 

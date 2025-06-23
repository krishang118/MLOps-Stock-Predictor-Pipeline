from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import pandas as pd
import joblib
from .utils import load_model, calculate_technical_indicators
import logging
import os
import time
import json
from datetime import datetime
import mlflow
import pickle
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import numpy as np
import re
logger = logging.getLogger(__name__)
model = None
scaler = None
feature_columns = None
loaded_model_name = "None"
REGISTERED_MODEL_NAME = "stock-predictor-xgb"
try:
    xgb_path = "models/xgboost_tuned_latest.pkl"
    if os.path.exists(xgb_path):
        model = load_model(xgb_path)
        loaded_model_name = f"Local file: {xgb_path}"
        logger.info(f"Loaded model from local file: {xgb_path}")
    else:
        raise FileNotFoundError("No local model file found.")
except Exception as local_e:
    logger.error(f"Failed to load model from local file: {local_e}")
    try:
        mlflow.set_tracking_uri("file:///Users/krishangsharma/Desktop/stock-predictor-mlops/mlruns")
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        loaded_model_name = f"MLflow Registry: {model_uri}"
        logger.info(f"Loaded model from MLflow Registry: {model_uri}")
    except Exception as mlflow_e:
        logger.error(f"Failed to load model from MLflow Registry as well: {mlflow_e}")
try:
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    logger.info("Preprocessing objects (scaler, columns) loaded successfully.")
except Exception as e:
    logger.error(f"Error loading preprocessing objects: {e}")
    model = None
app = FastAPI(title="Stock Market Predictor API", version="1.0.0")
file_handler = logging.FileHandler('logs/api.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(file_handler)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = None
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        logger.info(f"{request.method} {request.url.path} status={response.status_code} latency={process_time:.2f}ms")
        return response
    except Exception as e:
        logger.error(f"Exception during request: {e}")
        raise
class StockData(BaseModel):
    symbol: str
    open_price: float
    high: float
    low: float
    close: float
    volume: int
    ma_5: float
    ma_10: float
    ma_20: float
    rsi: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_lower: float
    volume_ratio: float
    price_change: float
    high_low_pct: float
    open_close_pct: float
    close_lag_1: float
    close_lag_2: float
    close_lag_3: float
    close_lag_5: float
    volume_lag_1: int
    volume_lag_2: int
    volume_lag_3: int
    volume_lag_5: int
    day_of_week: int
    month: int
    quarter: int
    sp500_return: float
    qqq_return: float
    dia_return: float
    sp500_return_lag_1: float
    sp500_return_lag_2: float
    sp500_return_lag_3: float
    sp500_return_lag_5: float
    relative_strength: float
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    confidence: str
class SymbolRequest(BaseModel):
    symbol: str
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Prediction Tool</title>
        <style>
            body {
                font-family: -apple-system, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            .form-section {
                margin-bottom: 30px;
                padding: 20px;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                background: #f9f9f9;
            }
            .form-section h3 {
                margin-top: 0;
                color: #555;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }
            .form-row {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 15px;
            }
            .form-group {
                display: flex;
                flex-direction: column;
            }
            label {
                font-weight: bold;
                margin-bottom: 5px;
                color: #333;
            }
            input, select {
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
            }
            input:focus, select:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 5px rgba(102, 126, 234, 0.3);
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                transition: transform 0.2s;
                width: 100%;
                margin-top: 20px;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                border-radius: 10px;
                display: none;
            }
            .result.success {
                background: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
            }
            .result.error {
                background: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
            }
            .prediction-details {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            .prediction-item {
                background: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .prediction-value {
                font-size: 1.5em;
                font-weight: bold;
                margin-bottom: 5px;
            }
            .confidence-high { color: #28a745; }
            .confidence-medium { color: #ffc107; }
            .confidence-low { color: #dc3545; }
            .loading {
                text-align: center;
                padding: 20px;
                display: none;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .info-box {
                background: #e3f2fd;
                border: 1px solid #bbdefb;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
            }
            .info-box h4 {
                margin-top: 0;
                color: #1976d2;
            }
            .quick-prediction-section {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border: 2px solid #dee2e6;
                border-radius: 12px;
                padding: 25px;
                margin-bottom: 30px;
            }
            .quick-prediction-section h3 {
                color: #495057;
                margin-top: 0;
                margin-bottom: 10px;
            }
            .quick-prediction-section p {
                color: #6c757d;
                margin-bottom: 20px;
            }
            .quick-form {
                display: flex;
                gap: 15px;
                align-items: end;
                flex-wrap: wrap;
            }
            .quick-form .form-group {
                flex: 1;
                min-width: 200px;
            }
            .quick-form button {
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
                padding: 12px 20px;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                cursor: pointer;
                transition: transform 0.2s;
                white-space: nowrap;
                text-align: center;
                flex-grow: 1;
            }
            .quick-form button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
            }
            .quick-result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 8px;
                background: white;
                border: 1px solid #dee2e6;
            }
            .quick-result.success {
                background: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
            }
            .quick-result.error {
                background: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
            }
            .spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #28a745;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .prediction-details {
                display: flex;
                gap: 20px;
                margin: 15px 0;
                flex-wrap: wrap;
            }
            .prediction-item {
                text-align: center;
                flex: 1;
                min-width: 100px;
            }
            .prediction-item .prediction-value {
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 5px;
            }
            .prediction-item div:last-child {
                font-size: 12px;
                color: #6c757d;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Stock Market Prediction Tool</h1>
            <div class="info-box">
                <h4>How to use this tool:</h4>
                <p>Enter the current stock data and technical indicators. The ML model will predict whether the stock price is likely to go up (1) or down (0) in the next period.</p>
            </div>

            <!-- Quick Prediction Mode -->
            <div class="quick-prediction-section">
                <h3>Quick Prediction Mode</h3>
                <p>Get instant predictions with just a stock symbol using sample market data.</p>
                <div class="quick-form">
                    <div class="form-group">
                        <label for="quick_symbol">Stock Symbol:</label>
                        <input type="text" id="quick_symbol" placeholder="e.g., AAPL, MSFT, TSLA" value="AAPL">
                    </div>
                    <button type="button" id="quickPredictBtn">Get Quick Prediction (A Demo)</button>
                    <button type="button" id="realTimePredictBtn" style="background: linear-gradient(135deg, #ff9800 0%, #ff5722 100%); color: white;">Real-Time Prediction</button>
                </div>
                <div class="quick-result" id="quickResult" style="display: none;"></div>
                <div class="quick-result" id="realTimeResult" style="display: none;"></div>
            </div>

            <hr style="margin: 30px 0; border: none; border-top: 2px solid #e0e0e0;">

            <h3>Detailed Prediction Mode</h3>
            <p>For precise predictions, fill in all the technical indicators below:</p>

            <form id="predictionForm">
                <div class="form-section">
                    <h3>Basic Stock Information</h3>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="symbol">Stock Symbol:</label>
                            <input type="text" id="symbol" name="symbol" value="AAPL" required>
                        </div>
                        <div class="form-group">
                            <label for="open_price">Open Price ($):</label>
                            <input type="number" id="open_price" name="open_price" step="0.01" value="190.00" required>
                        </div>
                        <div class="form-group">
                            <label for="high">High Price ($):</label>
                            <input type="number" id="high" name="high" step="0.01" value="191.00" required>
                        </div>
                        <div class="form-group">
                            <label for="low">Low Price ($):</label>
                            <input type="number" id="low" name="low" step="0.01" value="175.00" required>
                        </div>
                        <div class="form-group">
                            <label for="close">Close Price ($):</label>
                            <input type="number" id="close" name="close" step="0.01" value="178.00" required>
                        </div>
                        <div class="form-group">
                            <label for="volume">Volume:</label>
                            <input type="number" id="volume" name="volume" value="4000000" required>
                        </div>
                    </div>
                </div>

                <div class="form-section">
                    <h3>Technical Indicators</h3>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="ma_5">5-Day Moving Average ($):</label>
                            <input type="number" id="ma_5" name="ma_5" step="0.01" value="182.00" required>
                        </div>
                        <div class="form-group">
                            <label for="ma_10">10-Day Moving Average ($):</label>
                            <input type="number" id="ma_10" name="ma_10" step="0.01" value="186.00" required>
                        </div>
                        <div class="form-group">
                            <label for="ma_20">20-Day Moving Average ($):</label>
                            <input type="number" id="ma_20" name="ma_20" step="0.01" value="192.00" required>
                        </div>
                        <div class="form-group">
                            <label for="rsi">RSI (0-100):</label>
                            <input type="number" id="rsi" name="rsi" step="0.1" value="20.0" min="0" max="100" required>
                        </div>
                        <div class="form-group">
                            <label for="macd">MACD:</label>
                            <input type="number" id="macd" name="macd" step="0.01" value="-2.50" required>
                        </div>
                        <div class="form-group">
                            <label for="macd_signal">MACD Signal:</label>
                            <input type="number" id="macd_signal" name="macd_signal" step="0.01" value="-1.20" required>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="bb_upper">Bollinger Band Upper ($):</label>
                            <input type="number" id="bb_upper" name="bb_upper" step="0.01" value="195.00" required>
                        </div>
                        <div class="form-group">
                            <label for="bb_lower">Bollinger Band Lower ($):</label>
                            <input type="number" id="bb_lower" name="bb_lower" step="0.01" value="175.00" required>
                        </div>
                        <div class="form-group">
                            <label for="volume_ratio">Volume Ratio:</label>
                            <input type="number" id="volume_ratio" name="volume_ratio" step="0.01" value="3.50" required>
                        </div>
                        <div class="form-group">
                            <label for="price_change">Price Change ($):</label>
                            <input type="number" id="price_change" name="price_change" step="0.01" value="-6.30" required>
                        </div>
                        <div class="form-group">
                            <label for="high_low_pct">High-Low %:</label>
                            <input type="number" id="high_low_pct" name="high_low_pct" step="0.01" value="9.10" required>
                        </div>
                        <div class="form-group">
                            <label for="open_close_pct">Open-Close %:</label>
                            <input type="number" id="open_close_pct" name="open_close_pct" step="0.01" value="-6.32" required>
                        </div>
                    </div>
                </div>

                <div class="form-section">
                    <h3>Historical Data & Market Context</h3>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="close_lag_1">Previous Close ($):</label>
                            <input type="number" id="close_lag_1" name="close_lag_1" step="0.01" value="184.00" required>
                        </div>
                        <div class="form-group">
                            <label for="close_lag_2">2 Days Ago Close ($):</label>
                            <input type="number" id="close_lag_2" name="close_lag_2" step="0.01" value="190.00" required>
                        </div>
                        <div class="form-group">
                            <label for="close_lag_3">3 Days Ago Close ($):</label>
                            <input type="number" id="close_lag_3" name="close_lag_3" step="0.01" value="195.00" required>
                        </div>
                        <div class="form-group">
                            <label for="close_lag_5">5 Days Ago Close ($):</label>
                            <input type="number" id="close_lag_5" name="close_lag_5" step="0.01" value="200.00" required>
                        </div>
                        <div class="form-group">
                            <label for="volume_lag_1">Previous Volume:</label>
                            <input type="number" id="volume_lag_1" name="volume_lag_1" value="2500000" required>
                        </div>
                        <div class="form-group">
                            <label for="volume_lag_2">2 Days Ago Volume:</label>
                            <input type="number" id="volume_lag_2" name="volume_lag_2" value="2000000" required>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="volume_lag_3">3 Days Ago Volume:</label>
                            <input type="number" id="volume_lag_3" name="volume_lag_3" value="1500000" required>
                        </div>
                        <div class="form-group">
                            <label for="volume_lag_5">5 Days Ago Volume:</label>
                            <input type="number" id="volume_lag_5" name="volume_lag_5" value="1200000" required>
                        </div>
                        <div class="form-group">
                            <label for="day_of_week">Day of Week (0=Monday):</label>
                            <select id="day_of_week" name="day_of_week" required>
                                <option value="0">Monday</option>
                                <option value="1">Tuesday</option>
                                <option value="2" selected>Wednesday</option>
                                <option value="3">Thursday</option>
                                <option value="4">Friday</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="month">Month (1-12):</label>
                            <input type="number" id="month" name="month" min="1" max="12" value="6" required>
                        </div>
                        <div class="form-group">
                            <label for="quarter">Quarter (1-4):</label>
                            <input type="number" id="quarter" name="quarter" min="1" max="4" value="2" required>
                        </div>
                    </div>
                </div>

                <div class="form-section">
                    <h3>Market Context</h3>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="sp500_return">S&P 500 Return (%):</label>
                            <input type="number" id="sp500_return" name="sp500_return" step="0.001" value="-0.040" required>
                        </div>
                        <div class="form-group">
                            <label for="qqq_return">QQQ Return (%):</label>
                            <input type="number" id="qqq_return" name="qqq_return" step="0.001" value="-0.045" required>
                        </div>
                        <div class="form-group">
                            <label for="dia_return">DIA Return (%):</label>
                            <input type="number" id="dia_return" name="dia_return" step="0.001" value="-0.035" required>
                        </div>
                        <div class="form-group">
                            <label for="sp500_return_lag_1">S&P 500 Return (1 day ago) (%):</label>
                            <input type="number" id="sp500_return_lag_1" name="sp500_return_lag_1" step="0.001" value="-0.035" required>
                        </div>
                        <div class="form-group">
                            <label for="sp500_return_lag_2">S&P 500 Return (2 days ago) (%):</label>
                            <input type="number" id="sp500_return_lag_2" name="sp500_return_lag_2" step="0.001" value="-0.030" required>
                        </div>
                        <div class="form-group">
                            <label for="sp500_return_lag_3">S&P 500 Return (3 days ago) (%):</label>
                            <input type="number" id="sp500_return_lag_3" name="sp500_return_lag_3" step="0.001" value="-0.025" required>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="sp500_return_lag_5">S&P 500 Return (5 days ago) (%):</label>
                            <input type="number" id="sp500_return_lag_5" name="sp500_return_lag_5" step="0.001" value="-0.020" required>
                        </div>
                        <div class="form-group">
                            <label for="relative_strength">Relative Strength:</label>
                            <input type="number" id="relative_strength" name="relative_strength" step="0.01" value="0.55" required>
                        </div>
                    </div>
                </div>

                <button type="submit">Get Prediction</button>
            </form>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing stock data...</p>
            </div>

            <div class="result" id="result"></div>
        </div>

        <script>
            const quickPredictBtn = document.getElementById('quickPredictBtn');
            const realTimePredictBtn = document.getElementById('realTimePredictBtn');
            const quickSymbolInput = document.getElementById('quick_symbol');
            const quickResultDiv = document.getElementById('quickResult');
            const realTimeResultDiv = document.getElementById('realTimeResult');
            
            realTimePredictBtn.addEventListener('click', () => {
                const symbol = document.getElementById('quick_symbol').value.toUpperCase();
                if (!symbol) {
                    alert('Please enter a stock symbol.');
                    return;
                }

                const resultDiv = document.getElementById('realTimeResult');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<div class="spinner"></div><p>Fetching real-time data...</p>';
                resultDiv.className = 'quick-result';

                fetch('/fetch_and_predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol })
                })
                .then(response => {
                    if (!response.ok) {
                        return response.text().then(text => { throw new Error(`Network error: ${text}`) });
                    }
                    return response.json();
                })
                .then(data => {
                    const predictionText = data.prediction === 1 ? 'UP' : 'DOWN';
                    const displayProbability = data.prediction === 1 ? data.probability : 1 - data.probability;
                    const confidenceClass = data.confidence.toLowerCase();

                    resultDiv.innerHTML = `
                        <h4>Real-Time Prediction for ${symbol}</h4>
                        <div class="prediction-details">
                            <div class="prediction-item">
                                <div class="prediction-value confidence-${confidenceClass}">${predictionText}</div>
                                <div>Prediction</div>
                            </div>
                            <div class="prediction-item">
                                <div class="prediction-value">${(displayProbability * 100).toFixed(1)}%</div>
                                <div>Confidence</div>
                            </div>
                            <div class="prediction-item">
                                <div class="prediction-value confidence-${confidenceClass}">${data.confidence.toUpperCase()}</div>
                                <div>Reliability</div>
                            </div>
                        </div>
                        <p><em>This prediction uses the latest real market data and technical indicators.</em></p>
                    `;
                    resultDiv.className = 'quick-result success';
                    resultDiv.style.display = 'block';
                })
                .catch(error => {
                    resultDiv.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
                    resultDiv.style.display = 'block';
                });
            });

            quickPredictBtn.addEventListener('click', async () => {
                const symbol = quickSymbolInput.value.toUpperCase();
                if (!symbol) {
                    alert('Please enter a stock symbol');
                    return;
                }
                
                const quickResult = document.getElementById('quickResult');
                quickResult.style.display = 'block';
                quickResult.innerHTML = '<div class="spinner"></div><p>Getting quick prediction...</p>';
                quickResult.className = 'quick-result';                
                function rand(min, max, decimals = 2) {
                    return +(Math.random() * (max - min) + min).toFixed(decimals);
                }
                function randInt(min, max) {
                    return Math.floor(Math.random() * (max - min + 1)) + min;
                }
                const now = new Date();
                const dayOfWeek = randInt(0, 4); 
                const month = randInt(1, 12);
                const quarter = Math.ceil(month / 3);
                const basePrice = rand(20, 500, 2);
                const sampleData = {
                    symbol: symbol,
                    open_price: basePrice,
                    high: basePrice + rand(0, 10),
                    low: basePrice - rand(0, 10),
                    close: basePrice + rand(-5, 5),
                    volume: randInt(100000, 10000000),
                    ma_5: basePrice + rand(-2, 2),
                    ma_10: basePrice + rand(-4, 4),
                    ma_20: basePrice + rand(-8, 8),
                    rsi: rand(10, 90),
                    macd: rand(-3, 3),
                    macd_signal: rand(-3, 3),
                    bb_upper: basePrice + rand(5, 20),
                    bb_lower: basePrice - rand(5, 20),
                    volume_ratio: rand(0.5, 4),
                    price_change: rand(-10, 10),
                    high_low_pct: rand(0.5, 15),
                    open_close_pct: rand(-5, 5),
                    close_lag_1: basePrice + rand(-10, 10),
                    close_lag_2: basePrice + rand(-15, 15),
                    close_lag_3: basePrice + rand(-20, 20),
                    close_lag_5: basePrice + rand(-25, 25),
                    volume_lag_1: randInt(100000, 10000000),
                    volume_lag_2: randInt(100000, 10000000),
                    volume_lag_3: randInt(100000, 10000000),
                    volume_lag_5: randInt(100000, 10000000),
                    day_of_week: dayOfWeek,
                    month: month,
                    quarter: quarter,
                    sp500_return: rand(-0.05, 0.05, 4),
                    qqq_return: rand(-0.05, 0.05, 4),
                    dia_return: rand(-0.05, 0.05, 4),
                    sp500_return_lag_1: rand(-0.05, 0.05, 4),
                    sp500_return_lag_2: rand(-0.05, 0.05, 4),
                    sp500_return_lag_3: rand(-0.05, 0.05, 4),
                    sp500_return_lag_5: rand(-0.05, 0.05, 4),
                    relative_strength: rand(-2, 2)
                };
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(sampleData)
                    });
                    
                    const prediction = await response.json();
                    
                    if (response.ok) {
                        const direction = prediction.prediction === 1 ? 'UP' : 'DOWN';
                        const confidenceClass = prediction.confidence === 'high' ? 'confidence-high' : 
                                               prediction.confidence === 'medium' ? 'confidence-medium' : 'confidence-low';
                        
                        quickResult.innerHTML = `
                            <h4>Quick Prediction for ${symbol}</h4>
                            <div class="prediction-details">
                                <div class="prediction-item">
                                    <div class="prediction-value ${confidenceClass}">${direction}</div>
                                    <div>Prediction</div>
                                </div>
                                <div class="prediction-item">
                                    <div class="prediction-value">
                                        ${(prediction.probability * 100).toFixed(1)}%</div>
                                    <div>Confidence</div>
                                </div>
                                <div class="prediction-item">
                                    <div class="prediction-value ${confidenceClass}">${prediction.confidence.toUpperCase()}</div>
                                    <div>Reliability</div>
                                </div>
                            </div>
                            <p><em>Note: This prediction uses dynamic sample market data. For more accurate predictions, use the detailed mode below.</em></p>
                        `;
                        quickResult.className = 'quick-result success';
                    } else {
                        quickResult.innerHTML = `<h4>Error</h4><p>${prediction.detail || 'An error occurred while processing your request.'}</p>`;
                        quickResult.className = 'quick-result error';
                    }
                } catch (error) {
                    quickResult.innerHTML = `<h4>Error</h4><p>Network error: ${error.message}</p>`;
                    quickResult.className = 'quick-result error';
                }
            });

            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                
                loading.style.display = 'block';
                result.style.display = 'none';
                
                const formData = new FormData(this);
                const data = {};
                
                for (let [key, value] of formData.entries()) {
                    data[key] = parseFloat(value) || value;
                }
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const prediction = await response.json();
                    
                    if (response.ok) {
                        const direction = prediction.prediction === 1 ? 'UP' : 'DOWN';
                        const confidenceClass = prediction.confidence === 'high' ? 'confidence-high' : 
                                               prediction.confidence === 'medium' ? 'confidence-medium' : 'confidence-low';
                        
                        result.innerHTML = `
                            <h3>Prediction Result for ${data.symbol}</h3>
                            <div class="prediction-details">
                                <div class="prediction-item">
                                    <div class="prediction-value ${confidenceClass}">${direction}</div>
                                    <div>Prediction</div>
                                </div>
                                <div class="prediction-item">
                                    <div class="prediction-value">${(prediction.probability * 100).toFixed(1)}%</div>
                                    <div>Confidence</div>
                                </div>
                                <div class="prediction-item">
                                    <div class="prediction-value ${confidenceClass}">${prediction.confidence.toUpperCase()}</div>
                                    <div>Reliability</div>
                                </div>
                            </div>
                            <p><strong>Interpretation:</strong> The model predicts the stock price will move ${prediction.prediction === 1 ? 'upward' : 'downward'} with ${(prediction.probability * 100).toFixed(1)}% confidence. This is a ${prediction.confidence} confidence prediction.</p>
                        `;
                        result.className = 'result success';
                    } else {
                        result.innerHTML = `<h3>Error</h3><p>${prediction.detail || 'An error occurred while processing your request.'}</p>`;
                        result.className = 'result error';
                    }
                } catch (error) {
                    result.innerHTML = `<h3>Error</h3><p>Network error: ${error.message}</p>`;
                    result.className = 'result error';
                } finally {
                    loading.style.display = 'none';
                    result.style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content
@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}
@app.get("/model_info")
async def model_info():
    return {"model_path": loaded_model_name, "features": feature_columns}
@app.post("/predict", response_model=PredictionResponse)
async def predict(stock_data: StockData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        input_dict = stock_data.dict()
        logger.info(f"/predict request: {json.dumps(input_dict)} at {datetime.now().isoformat()}")
        input_data = pd.DataFrame([{
            'Open': stock_data.open_price,
            'High': stock_data.high,
            'Low': stock_data.low,
            'Close': stock_data.close,
            'Volume': stock_data.volume,
            'MA_5': stock_data.ma_5,
            'MA_10': stock_data.ma_10,
            'MA_20': stock_data.ma_20,
            'RSI': stock_data.rsi,
            'MACD': stock_data.macd,
            'MACD_signal': stock_data.macd_signal,
            'BB_upper': stock_data.bb_upper,
            'BB_lower': stock_data.bb_lower,
            'Volume_ratio': stock_data.volume_ratio,
            'Price_change': stock_data.price_change,
            'High_Low_pct': stock_data.high_low_pct,
            'Open_Close_pct': stock_data.open_close_pct,
            'Close_lag_1': stock_data.close_lag_1,
            'Close_lag_2': stock_data.close_lag_2,
            'Close_lag_3': stock_data.close_lag_3,
            'Close_lag_5': stock_data.close_lag_5,
            'Volume_lag_1': stock_data.volume_lag_1,
            'Volume_lag_2': stock_data.volume_lag_2,
            'Volume_lag_3': stock_data.volume_lag_3,
            'Volume_lag_5': stock_data.volume_lag_5,
            'Day_of_week': stock_data.day_of_week,
            'Month': stock_data.month,
            'Quarter': stock_data.quarter,
            'SP500_return': stock_data.sp500_return,
            'QQQ_return': stock_data.qqq_return,
            'DIA_return': stock_data.dia_return,
            'SP500_return_lag_1': stock_data.sp500_return_lag_1,
            'SP500_return_lag_2': stock_data.sp500_return_lag_2,
            'SP500_return_lag_3': stock_data.sp500_return_lag_3,
            'SP500_return_lag_5': stock_data.sp500_return_lag_5,
            'Relative_strength': stock_data.relative_strength
        }])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        if probability > 0.8 or probability < 0.2:
            confidence = "high"
        elif probability > 0.6 or probability < 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        response = PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence
        )
        logger.info(f"/predict response: {response.dict()} at {datetime.now().isoformat()}")
        return response
    except Exception as e:
        logger.error(f"Prediction error: {str(e)} at {datetime.now().isoformat()}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
@app.post("/batch_predict")
async def batch_predict(stock_data_list: List[StockData]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        logger.info(f"/batch_predict request: {len(stock_data_list)} items at {datetime.now().isoformat()}")
        predictions = []
        for stock_data in stock_data_list:
            input_data = pd.DataFrame([{
                'Open': stock_data.open_price,
                'High': stock_data.high,
                'Low': stock_data.low,
                'Close': stock_data.close,
                'Volume': stock_data.volume,
                'MA_5': stock_data.ma_5,
                'MA_10': stock_data.ma_10,
                'MA_20': stock_data.ma_20,
                'RSI': stock_data.rsi,
                'MACD': stock_data.macd,
                'MACD_signal': stock_data.macd_signal,
                'BB_upper': stock_data.bb_upper,
                'BB_lower': stock_data.bb_lower,
                'Volume_ratio': stock_data.volume_ratio,
                'Price_change': stock_data.price_change,
                'High_Low_pct': stock_data.high_low_pct,
                'Open_Close_pct': stock_data.open_close_pct,
                'Close_lag_1': stock_data.close_lag_1,
                'Close_lag_2': stock_data.close_lag_2,
                'Close_lag_3': stock_data.close_lag_3,
                'Close_lag_5': stock_data.close_lag_5,
                'Volume_lag_1': stock_data.volume_lag_1,
                'Volume_lag_2': stock_data.volume_lag_2,
                'Volume_lag_3': stock_data.volume_lag_3,
                'Volume_lag_5': stock_data.volume_lag_5,
                'Day_of_week': stock_data.day_of_week,
                'Month': stock_data.month,
                'Quarter': stock_data.quarter,
                'SP500_return': stock_data.sp500_return,
                'QQQ_return': stock_data.qqq_return,
                'DIA_return': stock_data.dia_return,
                'SP500_return_lag_1': stock_data.sp500_return_lag_1,
                'SP500_return_lag_2': stock_data.sp500_return_lag_2,
                'SP500_return_lag_3': stock_data.sp500_return_lag_3,
                'SP500_return_lag_5': stock_data.sp500_return_lag_5,
                'Relative_strength': stock_data.relative_strength
            }])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            if probability > 0.8 or probability < 0.2:
                confidence = "high"
            elif probability > 0.6 or probability < 0.4:
                confidence = "medium"
            else:
                confidence = "low"
            predictions.append({
                "symbol": stock_data.symbol,
                "prediction": int(prediction),
                "probability": float(probability),
                "confidence": confidence
            })
        logger.info(f"/batch_predict response: {json.dumps(predictions)} at {datetime.now().isoformat()}")
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)} at {datetime.now().isoformat()}")
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")
@app.post("/fetch_and_predict", response_model=PredictionResponse)
async def fetch_and_predict(request: SymbolRequest):
    symbol = request.symbol
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not ALPHA_VANTAGE_API_KEY:
        raise HTTPException(status_code=500, detail="Alpha Vantage API key not set")
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, meta = ts.get_daily(symbol=symbol, outputsize='full')
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }, inplace=True)        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        data.dropna(inplace=True)
        data.sort_index(ascending=True, inplace=True)
        features_df = calculate_technical_indicators(data.copy())
        if features_df.empty:
            raise HTTPException(status_code=500, detail="Feature calculation resulted in empty data.")
        latest_data = features_df.iloc[-1:].copy()
        if feature_columns:
            for col in feature_columns:
                if col not in latest_data.columns:
                    latest_data[col] = 0
            latest_data = latest_data[feature_columns]
        else:
             raise HTTPException(status_code=500, detail="Model feature columns not loaded.")
        latest_data_scaled = scaler.transform(latest_data)
        prediction = model.predict(latest_data_scaled)[0]
        probability = model.predict_proba(latest_data_scaled)[0][1]
        if probability > 0.8 or probability < 0.2:
            confidence = "high"
        elif probability > 0.6 or probability < 0.4:
            confidence = "medium"
        else:
            confidence = "low"            
        response = PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence
        )
        logger.info(f"/fetch_and_predict response: {response.dict()} for {symbol} at {datetime.now().isoformat()}")
        return response
    except Exception as e:
        error_message = str(e)
        if 'ALPHA_VANTAGE_API_KEY' in globals() and ALPHA_VANTAGE_API_KEY in error_message:
            error_message = error_message.replace(ALPHA_VANTAGE_API_KEY, '[REDACTED]')
        error_message = re.sub(r'\b[A-Z0-9]{16}\b', '[REDACTED]', error_message)
        logger.error(f"fetch_and_predict error: {error_message}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching or predicting for {symbol}. Error: {error_message}")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional
import joblib
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def setup_directories():
    dirs = ['data/raw', 'data/processed', 'models', 'logs', 'notebooks']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    logger.info("Project directories created successfully")
def save_model(model, model_name: str, model_dir: str = "models"):
    model_dir = os.path.join(os.getcwd(), model_dir)
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"{model_name}_{timestamp}.pkl")
    joblib.dump(model, model_path)
    latest_path = os.path.join(model_dir, f"{model_name}_latest.pkl")
    joblib.dump(model, latest_path)
    logger.info(f"Model saved to {model_path}")
    return model_path
def load_model(model_path: str):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))    
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()    
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)    
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['Stoch_%K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()
    df['TR'] = np.maximum.reduce([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ])
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                         np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                         np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
    tr14 = df['TR'].rolling(window=14).sum()
    plus_dm14 = df['+DM'].rolling(window=14).sum()
    minus_dm14 = df['-DM'].rolling(window=14).sum()
    plus_di14 = 100 * (plus_dm14 / tr14)
    minus_di14 = 100 * (minus_dm14 / tr14)
    dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    df['ADX'] = dx.rolling(window=14).mean()
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cci_ma = tp.rolling(window=20).mean()
    cci_std = tp.rolling(window=20).std()
    df['CCI'] = (tp - cci_ma) / (0.015 * cci_std)
    highest_high = df['High'].rolling(window=14).max()
    lowest_low = df['Low'].rolling(window=14).min()
    df['Williams_%R'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
    df['Rolling_volatility_10'] = df['Close'].pct_change().rolling(window=10).std()
    df['Rolling_volatility_20'] = df['Close'].pct_change().rolling(window=20).std()
    for window in [5, 10, 20]:
        df[f'Close_mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'Close_std_{window}'] = df['Close'].rolling(window=window).std()
        df[f'Close_min_{window}'] = df['Close'].rolling(window=window).min()
        df[f'Close_max_{window}'] = df['Close'].rolling(window=window).max()
        df[f'Volume_mean_{window}'] = df['Volume'].rolling(window=window).mean()
        df[f'Volume_std_{window}'] = df['Volume'].rolling(window=window).std()
        df[f'Volume_min_{window}'] = df['Volume'].rolling(window=window).min()
        df[f'Volume_max_{window}'] = df['Volume'].rolling(window=window).max()
    return df

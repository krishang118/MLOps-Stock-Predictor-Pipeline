"""
import pytest
import pandas as pd
import numpy as np
from src.utils import setup_directories
from src.preprocess import DataPreprocessor
from src.train_model import ModelTrainer
@pytest.fixture
def sample_stock_data():
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(150, 250, 100),
        'Low': np.random.uniform(50, 150, 100),
        'Close': np.random.uniform(100, 200, 100),
        'Volume': np.random.randint(1000000, 10000000, 100),
        'Symbol': ['AAPL'] * 100
    })
    return data
@pytest.fixture
def preprocessor():
    return DataPreprocessor()
@pytest.fixture
def trainer():
    return ModelTrainer()
""" 

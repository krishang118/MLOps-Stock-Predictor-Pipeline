import pytest
from src.preprocess import DataPreprocessor
from src.train_model import ModelTrainer
from src.utils import calculate_technical_indicators
def test_calculate_technical_indicators(sample_stock_data):
    result = calculate_technical_indicators(sample_stock_data)    
    assert 'MA_5' in result.columns
    assert 'MA_10' in result.columns
    assert 'MA_20' in result.columns
    assert 'RSI' in result.columns
    assert 'MACD' in result.columns    
    assert not result['MA_5'].isna().all()
    assert not result['RSI'].isna().all()
def test_data_preprocessing(sample_stock_data, preprocessor):
    features = preprocessor.create_features(sample_stock_data)    
    assert 'Target' in features.columns    
    assert 'Close_lag_1' in features.columns
    assert 'Volume_lag_1' in features.columns    
    assert features['Target'].dtype in [int, bool]
def test_model_training(preprocessor, trainer, sample_stock_data):
    features = preprocessor.create_features(sample_stock_data)
    clean_data = preprocessor.clean_data(features)
    X_train, X_test, y_train, y_test, feature_cols = preprocessor.prepare_training_data(clean_data)    
    model, metrics = trainer.train_model('random_forest', X_train, y_train, X_test, y_test)    
    assert model is not None
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1
def test_api_prediction_format():
    sample_input = {
        "symbol": "AAPL",
        "open_price": 150.0,
        "high": 155.0,
        "low": 149.0,
        "close": 152.0,
        "volume": 1000000,
        "ma_5": 151.0,
        "ma_10": 150.5,
        "ma_20": 149.8,
        "rsi": 55.0,
        "macd": 0.5,
        "macd_signal": 0.3,
        "bb_upper": 153.0,
        "bb_lower": 148.0,
        "volume_ratio": 1.2,
        "price_change": 0.013,
        "high_low_pct": 0.039,
        "open_close_pct": 0.013,
        "close_lag_1": 150.0,
        "close_lag_2": 149.5,
        "close_lag_3": 151.0,
        "close_lag_5": 148.0,
        "volume_lag_1": 950000,
        "volume_lag_2": 1100000,
        "volume_lag_3": 800000,
        "volume_lag_5": 1200000,
        "day_of_week": 1,
        "month": 6,
        "quarter": 2
    }    
    required_fields = [
        'symbol', 'open_price', 'high', 'low', 'close', 'volume',
        'ma_5', 'ma_10', 'ma_20', 'rsi', 'macd', 'macd_signal'
    ]
    for field in required_fields:
        assert field in sample_input

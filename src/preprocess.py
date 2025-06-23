import pandas as pd
import numpy as np
import os
from .utils import calculate_technical_indicators
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
logger = logging.getLogger(__name__)
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()        
    def load_raw_data(self, filepath: str = "data/raw/raw_stock_data.csv") -> pd.DataFrame:
        if os.path.exists(filepath):
            data = pd.read_csv(filepath)
            data = data.rename(columns={
                'date': 'Date',
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            data['Date'] = pd.to_datetime(data['Date'])
            logger.info(f"Loaded {len(data)} records from {filepath}")
            return data
        else:
            raise FileNotFoundError(f"Raw data file not found at {filepath}")
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()        
        df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)        
        processed_data = []
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol].copy()
            symbol_data = calculate_technical_indicators(symbol_data)
            processed_data.append(symbol_data)
        df = pd.concat(processed_data, ignore_index=True)        
        df['Price_change'] = df.groupby('Symbol')['Close'].pct_change()
        df['High_Low_pct'] = (df['High'] - df['Low']) / df['Close']
        df['Open_Close_pct'] = (df['Close'] - df['Open']) / df['Open']        
        for lag in [1, 2, 3, 5]:
            df[f'Close_lag_{lag}'] = df.groupby('Symbol')['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df.groupby('Symbol')['Volume'].shift(lag)        
        next_return = df.groupby('Symbol')['Close'].shift(-1) / df['Close'] - 1
        df['Target'] = np.where(next_return > 0.01, 1,
                         np.where(next_return < -0.01, 0, np.nan))        
        df['Day_of_week'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        spy = df[df['Symbol'] == 'SPY'][['Date', 'Close']].copy().sort_values('Date')
        spy['SP500_return'] = spy['Close'].pct_change()
        for lag in [1, 2, 3, 5]:
            spy[f'SP500_return_lag_{lag}'] = spy['SP500_return'].shift(lag)
        spy = spy[['Date', 'SP500_return'] + [f'SP500_return_lag_{lag}' for lag in [1,2,3,5]]]
        qqq = df[df['Symbol'] == 'QQQ'][['Date', 'Close']].copy().sort_values('Date')
        qqq['QQQ_return'] = qqq['Close'].pct_change()
        qqq = qqq[['Date', 'QQQ_return']]
        dia = df[df['Symbol'] == 'DIA'][['Date', 'Close']].copy().sort_values('Date')
        dia['DIA_return'] = dia['Close'].pct_change()
        dia = dia[['Date', 'DIA_return']]        
        df = pd.merge(df, spy, on='Date', how='left')
        df = pd.merge(df, qqq, on='Date', how='left')
        df = pd.merge(df, dia, on='Date', how='left')
        df.loc[df['Symbol'] == 'SPY', 'SP500_return'] = np.nan
        df.loc[df['Symbol'] == 'QQQ', 'QQQ_return'] = np.nan
        df.loc[df['Symbol'] == 'DIA', 'DIA_return'] = np.nan
        for lag in [1,2,3,5]:
            df.loc[df['Symbol'] == 'SPY', f'SP500_return_lag_{lag}'] = np.nan
        df['Relative_strength'] = df['Price_change'] - df['SP500_return']        
        df = df.dropna(subset=['Target'])
        df['Target'] = df['Target'].astype(int)        
        feature_cols = df.columns.difference(['Target'])
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        return df    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()        
        df = df.dropna()        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'Target':
                mean = df[col].mean()
                std = df[col].std()
                df = df[abs(df[col] - mean) <= (3 * std)]
        logger.info(f"Data cleaned, {len(df)} records remaining")
        return df    
    def prepare_training_data(self, data: pd.DataFrame):
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'RSI', 'MACD', 'MACD_signal',
            'BB_upper', 'BB_lower', 'Volume_ratio',
            'Price_change', 'High_Low_pct', 'Open_Close_pct',
            'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_5',
            'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3', 'Volume_lag_5',
            'Day_of_week', 'Month', 'Quarter', 'SP500_return', 'QQQ_return', 'DIA_return',
            'SP500_return_lag_1', 'SP500_return_lag_2', 'SP500_return_lag_3', 'SP500_return_lag_5',
            'Relative_strength'
        ]
        X = data[feature_columns]
        y = data['Target']
        X = X.sort_index()
        y = y.sort_index()
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X))
        train_idx, test_idx = splits[-1]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns    
    def save_processed_data(self, data: pd.DataFrame, filename: str = "processed_stock_data.csv"):
        filepath = os.path.join("data/processed", filename)
        data.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")
        return filepath
def walk_forward_splits(X, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(X):
        yield train_idx, test_idx
def main():
    preprocessor = DataPreprocessor()    
    raw_data = preprocessor.load_raw_data()
    processed_data = preprocessor.create_features(raw_data)
    cleaned_data = preprocessor.clean_data(processed_data)    
    preprocessor.save_processed_data(cleaned_data)    
    X_train, X_test, y_train, y_test, features = preprocessor.prepare_training_data(cleaned_data)
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info("Data preprocessing completed successfully")
if __name__ == "__main__":
    main()
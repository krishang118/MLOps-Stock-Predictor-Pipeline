from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
from typing import List
import pandas as pd
import os
from .utils import setup_directories
import logging
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY environment variable is required")
class DataIngester:
    def __init__(self, symbols: List[str], outputsize: str = "full"):
        self.symbols = symbols
        self.outputsize = outputsize
        self.ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    def fetch_stock_data(self, symbol: str) -> pd.DataFrame:
        try:
            data, meta = self.ts.get_daily(symbol=symbol, outputsize=self.outputsize)
            data['Symbol'] = symbol
            data.reset_index(inplace=True)
            logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()    
    def fetch_all_data(self) -> pd.DataFrame:
        all_data = []
        for symbol in self.symbols:
            data = self.fetch_stock_data(symbol)
            if not data.empty:
                all_data.append(data)
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined dataset has {len(combined_data)} records")
            return combined_data
        else:
            logger.error("No data fetched for any symbol")
            return pd.DataFrame()    
    def save_raw_data(self, data: pd.DataFrame, filename: str = "raw_stock_data.csv"):
        filepath = os.path.join("data/raw", filename)
        data.to_csv(filepath, index=False)
        logger.info(f"Raw data saved to {filepath}")
        return filepath
def main():
    setup_directories()    
    symbols = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'JPM', 'BAC', 'V', 'UNH', 'XOM',
        'SPY', 'QQQ', 'DIA', 'IWM', 'IVV', 'VOO', 'VTI', '^GSPC', '^IXIC', '^DJI']
    ingester = DataIngester(symbols, outputsize="full")
    raw_data = ingester.fetch_all_data()
    if not raw_data.empty:
        ingester.save_raw_data(raw_data)
        logger.info("Data ingestion completed successfully")
    else:
        logger.error("Data ingestion failed")
if __name__ == "__main__":
    main() 
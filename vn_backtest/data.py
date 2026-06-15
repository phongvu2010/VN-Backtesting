import os
import pandas as pd
from datetime import datetime
from vnstock import Market

class VNStockDataLoader:
    """
    Data loader for fetching and caching Vietnamese stock and index data.
    """
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = cache_dir
        self.market = Market()
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, symbol: str, start_date: str, end_date: str, is_index: bool) -> str:
        prefix = "index_" if is_index else "equity_"
        # Sanitize dates for filename
        start_clean = start_date.replace("-", "")
        end_clean = end_date.replace("-", "")
        return os.path.join(self.cache_dir, f"{prefix}{symbol}_{start_clean}_{end_clean}.csv")

    def fetch_data(self, symbol: str, start_date: str, end_date: str, is_index: bool = False, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch OHLCV data for a stock or index, caching it to CSV.
        
        Args:
            symbol (str): Ticker symbol (e.g. 'FPT', 'HPG', 'VNINDEX')
            start_date (str): Start date string (YYYY-MM-DD)
            end_date (str): End date string (YYYY-MM-DD)
            is_index (bool): True if downloading an index like VNINDEX
            use_cache (bool): If True, loads from cache if available
            
        Returns:
            pd.DataFrame: DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                          indexed by Date.
        """
        symbol = symbol.upper()
        cache_path = self._get_cache_path(symbol, start_date, end_date, is_index)
        
        if use_cache and os.path.exists(cache_path):
            df = pd.read_csv(cache_path, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            return df
            
        # If cache is not found or not used, fetch from API
        try:
            if is_index:
                # Fetch index data
                raw_df = self.market.index(symbol).ohlcv(
                    start=start_date, 
                    end=end_date, 
                    resolution='1D', 
                    count=10000
                )
            else:
                # Fetch stock data
                raw_df = self.market.equity(symbol).ohlcv(
                    start=start_date, 
                    end=end_date, 
                    resolution='1D', 
                    count=10000
                )
                
            if raw_df is None or raw_df.empty:
                raise ValueError(f"No data returned for {symbol} between {start_date} and {end_date}")
                
            # Standardize columns
            # Raw columns: ['time', 'open', 'high', 'low', 'close', 'volume']
            df = raw_df.copy()
            df['Date'] = pd.to_datetime(df['time'])
            
            # Map columns to Standard Title Case
            rename_map = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            df.rename(columns=rename_map, inplace=True)
            
            # Drop raw 'time' column and other unnecessary columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Multiply stock prices by 1000 to convert to VND (indices remain in points)
            if not is_index:
                for col in ['Open', 'High', 'Low', 'Close']:
                    df[col] = df[col] * 1000.0
            
            # Sort chronologically
            df.sort_values('Date', inplace=True)
            df.drop_duplicates(subset=['Date'], keep='first', inplace=True)
            
            # Save to cache
            df.to_csv(cache_path, index=False)
            
            df.set_index('Date', inplace=True)
            return df
            
        except Exception as e:
            raise RuntimeError(f"Error fetching data for {symbol} from vnstock: {e}")

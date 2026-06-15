import os
import pandas as pd
from datetime import datetime
from vnstock import Market
from vnstock.api.company import Company

class VNStockDataLoader:
    """
    Data loader for fetching and caching Vietnamese stock and index data.
    """
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = cache_dir
        self.market = Market()
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, symbol: str, is_index: bool) -> str:
        prefix = "index_" if is_index else "equity_"
        return os.path.join(self.cache_dir, f"{prefix}{symbol}.csv")

    def fetch_data(self, symbol: str, start_date: str, end_date: str, is_index: bool = False, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch OHLCV data for a stock or index, caching the entire history to a single CSV file.
        Applies a 12-hour TTL cache expiration to fetch fresh data if cached data is old.
        """
        symbol = symbol.upper()
        cache_path = self._get_cache_path(symbol, is_index)
        
        # Check cache freshness (12 hours TTL for price data)
        is_fresh = False
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            age_hours = (datetime.now().timestamp() - mtime) / 3600.0
            if age_hours < 12.0:
                is_fresh = True
                
        if use_cache and is_fresh:
            try:
                df = pd.read_csv(cache_path, parse_dates=['Date'])
                df.set_index('Date', inplace=True)
                # Slice the requested date range
                sliced_df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
                if not sliced_df.empty:
                    return sliced_df
            except Exception as e:
                print(f"CẢNH BÁO: Lỗi đọc cache cho {symbol} ({e}). Sẽ tải mới.")
                
        # If cache is not found, not fresh, or not used, fetch full history from API
        try:
            api_start = "2000-01-01"
            api_end = datetime.now().strftime("%Y-%m-%d")
            
            if is_index:
                raw_df = self.market.index(symbol).ohlcv(
                    start=api_start, 
                    end=api_end, 
                    resolution='1D', 
                    count=20000
                )
            else:
                raw_df = self.market.equity(symbol).ohlcv(
                    start=api_start, 
                    end=api_end, 
                    resolution='1D', 
                    count=20000
                )
                
            if raw_df is None or raw_df.empty:
                raise ValueError(f"No data returned for {symbol}")
                
            df = raw_df.copy()
            df['Date'] = pd.to_datetime(df['time'])
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)
            
            rename_map = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'average': 'Average'
            }
            df.rename(columns=rename_map, inplace=True)
            
            cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if 'Average' in df.columns:
                cols_to_keep.append('Average')
            df = df[cols_to_keep]
            
            if not is_index:
                cols_to_scale = ['Open', 'High', 'Low', 'Close']
                if 'Average' in df.columns:
                    cols_to_scale.append('Average')
                for col in cols_to_scale:
                    df[col] = df[col] * 1000.0
            
            df.sort_values('Date', inplace=True)
            df.drop_duplicates(subset=['Date'], keep='first', inplace=True)
            
            # Save the full history to cache
            df.to_csv(cache_path, index=False)
            
            df.set_index('Date', inplace=True)
            
            # Slice the requested date range
            sliced_df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
            return sliced_df
            
        except Exception as e:
            # Fallback to existing stale cache if API fails
            if os.path.exists(cache_path):
                print(f"CẢNH BÁO: Không thể tải dữ liệu mới cho {symbol} ({e}). Sử dụng dữ liệu cache cũ.")
                try:
                    df = pd.read_csv(cache_path, parse_dates=['Date'])
                    df.set_index('Date', inplace=True)
                    sliced_df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
                    if not sliced_df.empty:
                        return sliced_df
                except Exception:
                    pass
            raise RuntimeError(f"Error fetching data for {symbol} from vnstock: {e}")

    def _get_events_cache_path(self, symbol: str) -> str:
        return os.path.join(self.cache_dir, f"events_{symbol}.csv")

    def fetch_corporate_actions(self, symbol: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch corporate actions (dividends and splits) for a stock, caching it to CSV.
        Applies a 24-hour TTL cache expiration to update events.
        """
        symbol = symbol.upper()
        cache_path = self._get_events_cache_path(symbol)
        
        cols = ['exright_date', 'payout_date', 'value_per_share', 'exercise_ratio', 'listing_date', 'event_name_vi', 'event_title_vi']
        
        # Check cache freshness (24 hours TTL for events)
        is_fresh = False
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            age_hours = (datetime.now().timestamp() - mtime) / 3600.0
            if age_hours < 24.0:
                is_fresh = True
                
        if use_cache and is_fresh:
            try:
                df = pd.read_csv(cache_path, parse_dates=['exright_date', 'payout_date', 'listing_date'])
                return df
            except Exception as e:
                print(f"CẢNH BÁO: Lỗi đọc cache sự kiện cho {symbol} ({e}). Sẽ tải mới.")
                
        try:
            c = Company(symbol=symbol, source='VCI')
            raw_df = c.events()
            
            if raw_df is None or raw_df.empty:
                return pd.DataFrame(columns=cols)
                
            df = raw_df[raw_df['category'] == 'DIVIDEND'].copy()
            
            if df.empty:
                return pd.DataFrame(columns=cols)
                
            for col in cols:
                if col not in df.columns:
                    df[col] = None
                    
            df = df[cols].copy()
            
            for col in ['exright_date', 'payout_date', 'listing_date']:
                df[col] = pd.to_datetime(df[col])
                try:
                    df[col] = df[col].dt.tz_localize(None)
                except Exception:
                    pass
                
            df.to_csv(cache_path, index=False)
            return df
        except Exception as e:
            # Fallback to existing stale cache if API fails
            if os.path.exists(cache_path):
                print(f"CẢNH BÁO: Không thể lấy sự kiện doanh nghiệp mới cho {symbol} ({e}). Sử dụng cache cũ.")
                try:
                    df = pd.read_csv(cache_path, parse_dates=['exright_date', 'payout_date', 'listing_date'])
                    return df
                except Exception:
                    pass
            print(f"CẢNH BÁO: Không thể lấy sự kiện doanh nghiệp cho {symbol} ({e}).")
            return pd.DataFrame(columns=cols)

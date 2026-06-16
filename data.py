import os
import pandas as pd
from datetime import datetime
from vnstock import Market
from vnstock.api.company import Company

class VNStockDataLoader:
    """
    Data loader for fetching and caching Vietnamese stock and index data.
    """
    def __init__(self, cache_dir: str = "data_cache", local_db_dir: str = "local_data", offline: bool = False):
        self.cache_dir = cache_dir
        self.local_db_dir = local_db_dir
        self.offline = offline
        self.market = Market()
        os.makedirs(self.cache_dir, exist_ok=True)
        if self.local_db_dir:
            os.makedirs(self.local_db_dir, exist_ok=True)

    def _get_cache_path(self, symbol: str, is_index: bool) -> str:
        prefix = "index_" if is_index else "equity_"
        return os.path.join(self.cache_dir, f"{prefix}{symbol}.csv")

    def _standardize_and_scale(self, df: pd.DataFrame, symbol: str, is_index: bool) -> pd.DataFrame:
        df = df.copy()

        # Ensure index or 'Date' column is parsed
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif 'time' in df.columns:
            df['Date'] = pd.to_datetime(df['time'])
            df.set_index('Date', inplace=True)
        elif df.index.name != 'Date':
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'

        # Strip timezone if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Standardize column names
        rename_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'average': 'Average',
            'time': 'Date'
        }
        df.rename(columns=rename_map, inplace=True)

        # Keep only required columns
        cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'Average' in df.columns:
            cols_to_keep.append('Average')

        df = df[[col for col in cols_to_keep if col in df.columns]]

        # Scale to raw VND if not index
        if not is_index:
            cols_to_scale = ['Open', 'High', 'Low', 'Close']
            if 'Average' in df.columns:
                cols_to_scale.append('Average')
 
            # Dynamic threshold boundary:
            # Stocks (3 chars) -> threshold = 500.0
            # Warrants (8 chars) -> threshold = 10.0
            threshold = 10.0 if len(symbol) == 8 else 500.0

            if 'Close' in df.columns and df['Close'].max() < threshold:
                for col in cols_to_scale:
                    if col in df.columns:
                        df[col] = df[col].astype(float) * 1000.0

        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        return df

    def fetch_data(self, symbol: str, start_date: str, end_date: str, is_index: bool = False, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch OHLCV data for a stock or index, caching the entire history to a single CSV file.
        Applies local database loading, smart age-based updating, and API caching.
        """
        symbol = symbol.upper()
        cache_path = self._get_cache_path(symbol, is_index)
        
        # Helper to ensure index is timezone-naive DatetimeIndex
        def ensure_tz_naive(dataframe: pd.DataFrame) -> pd.DataFrame:
            if dataframe is not None and not dataframe.empty:
                if not isinstance(dataframe.index, pd.DatetimeIndex):
                    dataframe.index = pd.to_datetime(dataframe.index)
                if dataframe.index.tz is not None:
                    dataframe.index = dataframe.index.tz_localize(None)
            return dataframe
        
        # 1. Check local offline database first
        df_local = None
        if self.local_db_dir and os.path.exists(self.local_db_dir):
            csv_path = os.path.join(self.local_db_dir, f"{symbol}.csv")
            parquet_path = os.path.join(self.local_db_dir, f"{symbol}.parquet")

            if os.path.exists(parquet_path):
                try:
                    df_raw = pd.read_parquet(parquet_path)
                    df_local = self._standardize_and_scale(df_raw, symbol, is_index)
                    print(f"-> Đã tìm thấy tệp Parquet cục bộ: {parquet_path}")
                except Exception as e:
                    print(f"CẢNH BÁO: Không thể đọc tệp Parquet cục bộ {parquet_path} ({e})")
            elif os.path.exists(csv_path):
                try:
                    df_raw = pd.read_csv(csv_path)
                    df_local = self._standardize_and_scale(df_raw, symbol, is_index)
                    print(f"-> Đã tìm thấy tệp CSV cục bộ: {csv_path}")
                except Exception as e:
                    print(f"CẢNH BÁO: Không thể đọc tệp CSV cục bộ {csv_path} ({e})")

        # 1.5 Handle offline mode immediately
        if self.offline:
            if df_local is not None and not df_local.empty:
                df_local.to_csv(cache_path, index=True)
                return ensure_tz_naive(df_local.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)])
            if os.path.exists(cache_path):
                try:
                    df = pd.read_csv(cache_path, parse_dates=['Date'])
                    df.set_index('Date', inplace=True)
                    df = ensure_tz_naive(df)
                    sliced_df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
                    if not sliced_df.empty:
                        print(f"-> Chế độ Offline: Sử dụng dữ liệu cache cho {symbol}")
                        return sliced_df
                except Exception as e:
                    print(f"CẢNH BÁO: Lỗi đọc cache offline cho {symbol} ({e})")
            raise RuntimeError(f"Chế độ Offline: Không tìm thấy dữ liệu cho {symbol} tại {self.local_db_dir} hoặc cache.")

        # 2. Check if local database is too old and needs updating (only in online mode)
        if df_local is not None and not df_local.empty:
            last_date = df_local.index[-1]
            days_diff = (datetime.now().date() - last_date.date()).days

            # If the data is more than 30 days old, attempt incremental update from API
            if days_diff > 30:
                print(f"   Dữ liệu cục bộ của {symbol} đã cũ ({last_date.strftime('%Y-%m-%d')}). Đang tự động tải bù từ API...")
                try:
                    api_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    api_end = datetime.now().strftime("%Y-%m-%d")
                    
                    if is_index:
                        raw_df = self.market.index(symbol).ohlcv(
                            start=api_start, 
                            end=api_end, 
                            resolution='1D', 
                            count=5000
                        )
                    else:
                        raw_df = self.market.equity(symbol).ohlcv(
                            start=api_start, 
                            end=api_end, 
                            resolution='1D', 
                            count=5000
                        )
                    
                    if raw_df is not None and not raw_df.empty:
                        df_api = self._standardize_and_scale(raw_df, symbol, is_index)
                        # Merge local data with new API data
                        df_merged = pd.concat([df_local, df_api])
                        df_merged = df_merged[~df_merged.index.duplicated(keep='last')]
                        df_merged.sort_index(inplace=True)
                        
                        # Save the updated data to cache
                        df_merged.to_csv(cache_path, index=True)
                        print(f"   Cập nhật dữ liệu cho {symbol} thành công. Đã lưu vào bộ nhớ cache.")
                        
                        # Return the sliced requested range
                        return ensure_tz_naive(df_merged.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)])
                except Exception as e:
                    print(f"CẢNH BÁO: Không thể tải bù dữ liệu cho {symbol} ({e}). Tiếp tục chạy với dữ liệu cục bộ.")
            
            # Save local to cache and return sliced range
            df_local.to_csv(cache_path, index=True)
            return ensure_tz_naive(df_local.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)])

        # 3. Check cache freshness (12 hours TTL for price data)
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
                df = ensure_tz_naive(df)
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
                
            df = self._standardize_and_scale(raw_df, symbol, is_index)
            
            # Save the full history to cache
            df.to_csv(cache_path, index=True)
            
            # Slice the requested date range
            sliced_df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
            return ensure_tz_naive(sliced_df)
            
        except Exception as e:
            # Fallback to existing stale cache if API fails
            if os.path.exists(cache_path):
                print(f"CẢNH BÁO: Không thể tải dữ liệu mới cho {symbol} ({e}). Sử dụng dữ liệu cache cũ.")
                try:
                    df = pd.read_csv(cache_path, parse_dates=['Date'])
                    df.set_index('Date', inplace=True)
                    df = ensure_tz_naive(df)
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
        
        # Helper to ensure naive timestamps in date columns
        def ensure_cols_tz_naive(df_to_clean: pd.DataFrame) -> pd.DataFrame:
            if df_to_clean is not None and not df_to_clean.empty:
                for col in ['exright_date', 'payout_date', 'listing_date']:
                    if col in df_to_clean.columns:
                        df_to_clean[col] = pd.to_datetime(df_to_clean[col])
                        try:
                            # Use .dt accessor if it's a series of Datetime
                            if hasattr(df_to_clean[col], 'dt'):
                                df_to_clean[col] = df_to_clean[col].dt.tz_localize(None)
                            else:
                                df_to_clean[col] = df_to_clean[col].tz_localize(None)
                        except Exception:
                            pass
            return df_to_clean

        # 1. Check local offline database first
        df_local = None
        if self.local_db_dir and os.path.exists(self.local_db_dir):
            local_event_path = os.path.join(self.local_db_dir, f"events_{symbol}.csv")
            if os.path.exists(local_event_path):
                try:
                    df_raw = pd.read_csv(local_event_path, parse_dates=['exright_date', 'payout_date', 'listing_date'])
                    df_local = ensure_cols_tz_naive(df_raw)
                    print(f"-> Đã tìm thấy tệp sự kiện cục bộ: {local_event_path}")
                except Exception as e:
                    print(f"CẢNH BÁO: Không thể đọc tệp sự kiện cục bộ {local_event_path} ({e})")

        # 1.5 Handle offline mode immediately
        if self.offline:
            if df_local is not None and not df_local.empty:
                df_local.to_csv(cache_path, index=False)
                return df_local
            if os.path.exists(cache_path):
                try:
                    df = pd.read_csv(cache_path, parse_dates=['exright_date', 'payout_date', 'listing_date'])
                    return ensure_cols_tz_naive(df)
                except Exception:
                    pass
            return pd.DataFrame(columns=cols)

        # 2. Check cache freshness (24 hours TTL for events, only in online mode)
        is_fresh = False
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            age_hours = (datetime.now().timestamp() - mtime) / 3600.0
            if age_hours < 24.0:
                is_fresh = True
                
        if use_cache and is_fresh:
            try:
                df = pd.read_csv(cache_path, parse_dates=['exright_date', 'payout_date', 'listing_date'])
                return ensure_cols_tz_naive(df)
            except Exception as e:
                print(f"CẢNH BÁO: Lỗi đọc cache sự kiện cho {symbol} ({e}). Sẽ tải mới.")
                
        # 3. Fetch from API
        try:
            c = Company(symbol=symbol, source='VCI')
            raw_df = c.events()
            
            if raw_df is None or raw_df.empty:
                if df_local is not None and not df_local.empty:
                    df_local.to_csv(cache_path, index=False)
                    return df_local
                return pd.DataFrame(columns=cols)
                
            df = raw_df[raw_df['category'] == 'DIVIDEND'].copy()
            
            if df.empty:
                if df_local is not None and not df_local.empty:
                    df_local.to_csv(cache_path, index=False)
                    return df_local
                return pd.DataFrame(columns=cols)
                
            for col in cols:
                if col not in df.columns:
                    df[col] = None
                    
            df = df[cols].copy()
            df = ensure_cols_tz_naive(df)
                
            # Save to cache
            df.to_csv(cache_path, index=False)
            
            # Write backup to local database if possible
            if self.local_db_dir and os.path.exists(self.local_db_dir):
                local_event_path = os.path.join(self.local_db_dir, f"events_{symbol}.csv")
                df.to_csv(local_event_path, index=False)
                
            return df
        except Exception as e:
            # Fallback to local database or stale cache if API fails
            print("=" * 80)
            print(f"CẢNH BÁO: Không thể kết nối API vnstock để lấy sự kiện doanh nghiệp của {symbol}.")
            print("API có thể đã bị thay đổi, chặn, hoặc thiết bị mất kết nối mạng.")
            print(f"Chi tiết lỗi: {e}")
            print("Tự động chuyển sang sử dụng dữ liệu sự kiện backup/cache cục bộ...")
            print("=" * 80)
            
            if df_local is not None and not df_local.empty:
                try:
                    df_local.to_csv(cache_path, index=False)
                    return df_local
                except Exception:
                    pass
                    
            if os.path.exists(cache_path):
                try:
                    df = pd.read_csv(cache_path, parse_dates=['exright_date', 'payout_date', 'listing_date'])
                    return ensure_cols_tz_naive(df)
                except Exception:
                    pass
            return pd.DataFrame(columns=cols)

    def fetch_exchange_map(self, use_cache: bool = True) -> dict[str, str]:
        """
        Fetch exchange mapping for all symbols and cache it locally to JSON.
        TTL of 7 days to keep it updated while avoiding daily network costs.
        """
        import json
        cache_path = os.path.join(self.cache_dir, "exchange_map.json")
        
        # Check cache freshness (7 days TTL = 168 hours)
        is_fresh = False
        if os.path.exists(cache_path):
            try:
                mtime = os.path.getmtime(cache_path)
                age_hours = (datetime.now().timestamp() - mtime) / 3600.0
                if age_hours < 168.0:
                    is_fresh = True
            except Exception:
                pass
                
        # Handle offline mode immediately
        if self.offline:
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception:
                    pass
            return {
                'FPT': 'hose', 'HPG': 'hose', 'VNM': 'hose', 'VIC': 'hose', 'VHM': 'hose', 'TCB': 'hose',
                'MWG': 'hose', 'SSI': 'hose', 'VND': 'hose', 'VCB': 'hose', 'STB': 'hose', 'MBB': 'hose',
                'IDC': 'hnx', 'PVS': 'hnx', 'SHS': 'hnx', 'MBS': 'hnx', 'CEO': 'hnx', 'HUT': 'hnx',
                'BSR': 'upcom', 'ACV': 'upcom', 'VEA': 'upcom', 'VGI': 'upcom', 'QNS': 'upcom', 'LTG': 'upcom'
            }

        if use_cache and is_fresh:
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"CẢNH BÁO: Lỗi đọc cache sàn giao dịch ({e}). Sẽ tải mới.")
                
        # Fetch from vnstock Listing
        exchange_map = {}
        # Try multiple sources
        for source in ['KBS', 'VCI', 'MSN']:
            try:
                from vnstock import Listing
                l = Listing(source=source)
                
                # Call symbols_by_exchange once with correct arguments per source
                if source == 'VCI':
                    df_symbols = l.symbols_by_exchange(lang='vi')
                else:
                    df_symbols = l.symbols_by_exchange(get_all=True)
                    
                if df_symbols is not None and not df_symbols.empty and 'symbol' in df_symbols.columns and 'exchange' in df_symbols.columns:
                    df_symbols = df_symbols.dropna(subset=['symbol', 'exchange'])
                    for _, row in df_symbols.iterrows():
                        symbol = str(row['symbol']).upper()
                        exch = str(row['exchange']).lower().strip()
                        
                        if exch in ['hose', 'hsx']:
                            exch = 'hose'
                        elif exch in ['hnx', 'xhnf']:
                            exch = 'hnx'
                        elif exch in ['upcom', 'comup']:
                            exch = 'upcom'
                        else:
                            continue
                        exchange_map[symbol] = exch
                        
                    if exchange_map:
                        break
            except Exception as e:
                print(f"CẢNH BÁO: Lỗi lấy danh sách sàn từ nguồn {source}: {e}")
                
        # If successfully fetched, write to cache
        if exchange_map:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(exchange_map, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"CẢNH BÁO: Không thể ghi cache sàn giao dịch ({e})")
        else:
            # Fallback dictionary of popular stocks
            # Let's see if we can read from existing cache first even if stale
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        exchange_map = json.load(f)
                except Exception:
                    pass
            if not exchange_map:
                exchange_map = {
                    'FPT': 'hose', 'HPG': 'hose', 'VNM': 'hose', 'VIC': 'hose', 'VHM': 'hose', 'TCB': 'hose',
                    'MWG': 'hose', 'SSI': 'hose', 'VND': 'hose', 'VCB': 'hose', 'STB': 'hose', 'MBB': 'hose',
                    'IDC': 'hnx', 'PVS': 'hnx', 'SHS': 'hnx', 'MBS': 'hnx', 'CEO': 'hnx', 'HUT': 'hnx',
                    'BSR': 'upcom', 'ACV': 'upcom', 'VEA': 'upcom', 'VGI': 'upcom', 'QNS': 'upcom', 'LTG': 'upcom'
                }
                
        return exchange_map

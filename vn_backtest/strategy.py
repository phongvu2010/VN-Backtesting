import pandas as pd
import numpy as np
from typing import Callable, Any
import talib


class Strategy:
    """
    Base class for writing backtesting strategies.
    Inherit from this class and override init() and next() methods.
    """
    def __init__(self, data: Any, engine: Any):
        self.data = data
        self.engine = engine
        self._indicators = []
        self.current_idx = 0

    @property
    def current_time(self) -> pd.Timestamp:
        """Get current timestamp of the backtest simulation."""
        if hasattr(self.engine, 'dates'):
            return self.engine.dates[self.current_idx]
        return self.data.index[self.current_idx]

    @property
    def open(self) -> float:
        """Get the Open price of the current bar for the main ticker."""
        ticker = getattr(self.engine, 'main_ticker', list(self.data.keys())[0] if isinstance(self.data, dict) else None)
        if ticker:
            return self.get_open(ticker)
        return float(self.data['Open'].iloc[self.current_idx])

    @property
    def high(self) -> float:
        """Get the High price of the current bar for the main ticker."""
        ticker = getattr(self.engine, 'main_ticker', list(self.data.keys())[0] if isinstance(self.data, dict) else None)
        if ticker:
            return self.get_high(ticker)
        return float(self.data['High'].iloc[self.current_idx])

    @property
    def low(self) -> float:
        """Get the Low price of the current bar for the main ticker."""
        ticker = getattr(self.engine, 'main_ticker', list(self.data.keys())[0] if isinstance(self.data, dict) else None)
        if ticker:
            return self.get_low(ticker)
        return float(self.data['Low'].iloc[self.current_idx])

    @property
    def close(self) -> float:
        """Get the Close price of the current bar for the main ticker."""
        ticker = getattr(self.engine, 'main_ticker', list(self.data.keys())[0] if isinstance(self.data, dict) else None)
        if ticker:
            return self.get_close(ticker)
        return float(self.data['Close'].iloc[self.current_idx])

    @property
    def volume(self) -> float:
        """Get the Volume of the current bar for the main ticker."""
        ticker = getattr(self.engine, 'main_ticker', list(self.data.keys())[0] if isinstance(self.data, dict) else None)
        if ticker:
            return self.get_volume(ticker)
        return float(self.data['Volume'].iloc[self.current_idx])

    def get_open(self, ticker: str) -> float:
        """Get Open price of a specific ticker today."""
        df = self.data[ticker] if isinstance(self.data, dict) else self.data
        current_time = self.current_time
        if current_time in df.index:
            return float(df.loc[current_time, 'Open'])
        return float('nan')

    def get_high(self, ticker: str) -> float:
        """Get High price of a specific ticker today."""
        df = self.data[ticker] if isinstance(self.data, dict) else self.data
        current_time = self.current_time
        if current_time in df.index:
            return float(df.loc[current_time, 'High'])
        return float('nan')

    def get_low(self, ticker: str) -> float:
        """Get Low price of a specific ticker today."""
        df = self.data[ticker] if isinstance(self.data, dict) else self.data
        current_time = self.current_time
        if current_time in df.index:
            return float(df.loc[current_time, 'Low'])
        return float('nan')

    def get_close(self, ticker: str) -> float:
        """Get Close price of a specific ticker today."""
        df = self.data[ticker] if isinstance(self.data, dict) else self.data
        current_time = self.current_time
        if current_time in df.index:
            return float(df.loc[current_time, 'Close'])
        return float('nan')

    def get_volume(self, ticker: str) -> float:
        """Get Volume of a specific ticker today."""
        df = self.data[ticker] if isinstance(self.data, dict) else self.data
        current_time = self.current_time
        if current_time in df.index:
            return float(df.loc[current_time, 'Volume'])
        return float('nan')

    @property
    def cash(self) -> float:
        """Get total portfolio cash (settled + pending)."""
        return self.engine.cash

    @property
    def available_cash(self) -> float:
        """Get cash available to buy shares today."""
        return self.engine.available_cash

    @property
    def positions(self) -> dict:
        """Get total shares owned per ticker."""
        return self.engine.positions

    @property
    def sellable_shares(self) -> dict:
        """Get settled shares available to sell per ticker."""
        return self.engine.sellable_shares

    def init(self):
        """
        Initialize strategy indicators. 
        Override in subclass to precompute indicators on the historical data.
        """
        pass

    def next(self):
        """
        Define strategy logic for each trading day.
        Override in subclass. This is called on every trading day (bar).
        """
        pass

    def buy(self, ticker: str, size: float = None) -> None:
        """
        Place a Buy Order.
        
        Args:
            ticker (str): The ticker symbol to buy (e.g. 'FPT').
            size (float or int, optional): 
                - If float between 0.0 and 1.0 (e.g., 0.5): Allocates that percentage of available cash.
                - If integer > 1 (e.g., 200): Buys that exact number of shares.
                - If None: Allocates 100% of available cash.
        """
        self.engine.place_buy_order(ticker, size, time=self.current_time)

    def sell(self, ticker: str, size: float = None) -> None:
        """
        Place a Sell Order.
        
        Args:
            ticker (str): The ticker symbol to sell.
            size (float or int, optional):
                - If float between 0.0 and 1.0 (e.g., 0.5): Sells that percentage of the position.
                - If integer > 1: Sells that exact number of shares.
                - If None: Sells the entire position.
        """
        self.engine.place_sell_order(ticker, size, time=self.current_time)

    def I(self, func: Callable[..., pd.Series], *args, **kwargs) -> pd.Series:
        """
        Declare and compute an indicator.
        This will compute the indicator on the dataset at start.
        
        Can be called as:
            self.I(SMA, 20) -> computes SMA on Close of main ticker
            self.I(SMA, 'HPG', 20) -> computes SMA on Close of HPG
        """
        if args and isinstance(args[0], str) and isinstance(self.data, dict) and args[0] in self.data:
            ticker = args[0]
            func_args = args[1:]
            df = self.data[ticker]
        else:
            if isinstance(self.data, dict):
                ticker = getattr(self.engine, 'main_ticker', list(self.data.keys())[0])
                df = self.data[ticker]
            else:
                df = self.data
            func_args = args

        indicator_series = func(df, *func_args, **kwargs)
        self._indicators.append(indicator_series)
        return indicator_series

# --- Helper Technical Indicators ---

def SMA(data: pd.DataFrame, period: int = 20, column: str = 'Close') -> pd.Series:
    """Simple Moving Average using TA-Lib."""
    res = talib.SMA(data[column].to_numpy(dtype=float), timeperiod=period)
    return pd.Series(res, index=data.index)

def EMA(data: pd.DataFrame, period: int = 20, column: str = 'Close') -> pd.Series:
    """Exponential Moving Average using TA-Lib."""
    res = talib.EMA(data[column].to_numpy(dtype=float), timeperiod=period)
    return pd.Series(res, index=data.index)

def RSI(data: pd.DataFrame, period: int = 14, column: str = 'Close') -> pd.Series:
    """Relative Strength Index using TA-Lib (Wilder's Smoothing Average)."""
    res = talib.RSI(data[column].to_numpy(dtype=float), timeperiod=period)
    return pd.Series(res, index=data.index)

def MACD(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, column: str = 'Close') -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence using TA-Lib.
    Returns: (macd_line, signal_line, histogram)
    """
    macd_line, signal_line, histogram = talib.MACD(
        data[column].to_numpy(dtype=float),
        fastperiod=fast_period,
        slowperiod=slow_period,
        signalperiod=signal_period
    )
    return (
        pd.Series(macd_line, index=data.index),
        pd.Series(signal_line, index=data.index),
        pd.Series(histogram, index=data.index)
    )

def Bollinger_Bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0, column: str = 'Close') -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands using TA-Lib.
    Returns: (middle_band, upper_band, lower_band)
    """
    upper, middle, lower = talib.BBANDS(
        data[column].to_numpy(dtype=float),
        timeperiod=period,
        nbdevup=std_dev,
        nbdevdn=std_dev,
        matype=0  # 0 = SMA
    )
    return (
        pd.Series(middle, index=data.index),
        pd.Series(upper, index=data.index),
        pd.Series(lower, index=data.index)
    )


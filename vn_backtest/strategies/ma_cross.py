import pandas as pd
import numpy as np
from typing import Any
from ..strategy import Strategy, SMA

class MACrossover(Strategy):
    """
    Simple Moving Average Crossover Strategy.
    Buys when the fast SMA crosses above the slow SMA.
    Sells when the fast SMA crosses below the slow SMA.
    Works for both single-ticker and multi-ticker backtests.
    """
    def __init__(self, data: Any, engine):
        super().__init__(data, engine)
        self.fast_period = 10
        self.slow_period = 20
        # Determine tickers list
        if isinstance(self.data, dict):
            self.tickers = list(self.data.keys())
        else:
            self.tickers = [getattr(self.engine, 'ticker', 'FPT')]

    def init(self):
        # Precompute indicators for each ticker
        self.fast_mas = {}
        self.slow_mas = {}
        for ticker in self.tickers:
            # self.I knows how to handle ticker name as first argument
            self.fast_mas[ticker] = self.I(SMA, ticker, self.fast_period)
            self.slow_mas[ticker] = self.I(SMA, ticker, self.slow_period)

    def next(self):
        # Loop through all tickers in the portfolio
        for ticker in self.tickers:
            ticker_df = self.data[ticker] if isinstance(self.data, dict) else self.data
            current_time = self.current_time
            
            # Skip if ticker has no data on this day
            if current_time not in ticker_df.index:
                continue
                
            # Get integer index of current_time in ticker's DataFrame
            ticker_idx = ticker_df.index.get_loc(current_time)
            if ticker_idx < 1:
                continue

            fast_ma = self.fast_mas[ticker]
            slow_ma = self.slow_mas[ticker]

            # Access indicator values
            fast_curr = fast_ma.iloc[ticker_idx]
            fast_prev = fast_ma.iloc[ticker_idx - 1]
            slow_curr = slow_ma.iloc[ticker_idx]
            slow_prev = slow_ma.iloc[ticker_idx - 1]

            # Skip if indicators are NaN
            if pd.isna(fast_curr) or pd.isna(fast_prev) or pd.isna(slow_curr) or pd.isna(slow_prev):
                continue

            # Check positions for this ticker
            has_position = self.positions.get(ticker, 0) > 0

            # Crossover Up: Fast MA cuts above Slow MA -> BUY
            if fast_prev <= slow_prev and fast_curr > slow_curr:
                if not has_position:
                    # Allocate equal cash fraction for each stock in portfolio
                    buy_fraction = 1.0 / len(self.tickers)
                    self.buy(ticker, size=buy_fraction)

            # Crossover Down: Fast MA cuts below Slow MA -> SELL
            elif fast_prev >= slow_prev and fast_curr < slow_curr:
                if has_position:
                    self.sell(ticker)

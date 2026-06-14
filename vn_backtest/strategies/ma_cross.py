import pandas as pd
from ..strategy import Strategy, SMA

class MACrossover(Strategy):
    """
    Simple Moving Average Crossover Strategy.
    Buys when the fast SMA crosses above the slow SMA.
    Sells when the fast SMA crosses below the slow SMA.
    """
    def __init__(self, data: pd.DataFrame, engine):
        super().__init__(data, engine)
        self.fast_period = 10
        self.slow_period = 20
        self.ticker = getattr(self.engine, 'ticker', 'FPT')


    def init(self):
        # Declare indicators
        self.fast_ma = self.I(SMA, self.fast_period)
        self.slow_ma = self.I(SMA, self.slow_period)

    def next(self):
        # We need at least 2 bars to detect a crossover
        if self.current_idx < 1:
            return

        # Access indicator values
        fast_curr = self.fast_ma.iloc[self.current_idx]
        fast_prev = self.fast_ma.iloc[self.current_idx - 1]
        slow_curr = self.slow_ma.iloc[self.current_idx]
        slow_prev = self.slow_ma.iloc[self.current_idx - 1]

        # Skip if indicators are NaN
        if pd.isna(fast_curr) or pd.isna(fast_prev) or pd.isna(slow_curr) or pd.isna(slow_prev):
            return

        # Check positions for the ticker
        has_position = self.positions.get(self.ticker, 0) > 0

        # Crossover Up: Fast MA cuts above Slow MA -> BUY
        if fast_prev <= slow_prev and fast_curr > slow_curr:
            if not has_position:
                self.buy(self.ticker)

        # Crossover Down: Fast MA cuts below Slow MA -> SELL
        elif fast_prev >= slow_prev and fast_curr < slow_curr:
            if has_position:
                self.sell(self.ticker)

    def set_ticker(self, ticker: str):
        self.ticker = ticker

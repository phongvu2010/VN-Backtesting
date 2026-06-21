import pandas as pd
import talib

# --- Helper Technical Indicators ---


def SMA(data: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.Series:
    """Simple Moving Average using TA-Lib."""
    res = talib.SMA(data[column].to_numpy(dtype=float), timeperiod=period)
    return pd.Series(res, index=data.index)


def EMA(data: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.Series:
    """Exponential Moving Average using TA-Lib."""
    res = talib.EMA(data[column].to_numpy(dtype=float), timeperiod=period)
    return pd.Series(res, index=data.index)


def RSI(data: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.Series:
    """Relative Strength Index using TA-Lib (Wilder's Smoothing Average)."""
    res = talib.RSI(data[column].to_numpy(dtype=float), timeperiod=period)
    return pd.Series(res, index=data.index)


def MACD(
    data: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    column: str = "Close",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence using TA-Lib.
    Returns: (macd_line, signal_line, histogram)
    """
    macd_line, signal_line, histogram = talib.MACD(
        data[column].to_numpy(dtype=float),
        fastperiod=fast_period,
        slowperiod=slow_period,
        signalperiod=signal_period,
    )
    return (
        pd.Series(macd_line, index=data.index),
        pd.Series(signal_line, index=data.index),
        pd.Series(histogram, index=data.index),
    )


def Bollinger_Bands(
    data: pd.DataFrame, period: int = 20, std_dev: float = 2.0, column: str = "Close"
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands using TA-Lib.
    Returns: (middle_band, upper_band, lower_band)
    """
    upper, middle, lower = talib.BBANDS(
        data[column].to_numpy(dtype=float),
        timeperiod=period,
        nbdevup=std_dev,
        nbdevdn=std_dev,
        matype=0,  # 0 = SMA
    )
    return (
        pd.Series(middle, index=data.index),
        pd.Series(upper, index=data.index),
        pd.Series(lower, index=data.index),
    )

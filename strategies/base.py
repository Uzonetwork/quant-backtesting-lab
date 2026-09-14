"""
Abstract base class for all strategies.
Enforces the interface the backtest engine expects.
"""
from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """
    Subclasses implement generate_signals().
    Returns a DataFrame with columns:
      signal   : int  — +1 (long), -1 (short), 0 (flat / no signal)
      stop     : float — initial stop price for this bar's signal
      atr      : float — ATR value at signal time (used by sizing module)
    Index must match the input df index exactly.
    Signal on bar N means: enter at bar N+1 open (engine enforces this shift).
    """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : OHLCV DataFrame with UTC DatetimeIndex

        Returns
        -------
        signals : DataFrame[signal, stop, atr] — same index as df
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

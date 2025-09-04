# rsi.py
from dataclasses import dataclass
import pandas as pd

class Strategy:
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Minimal RSI implementation (Close-to-Close). Kept simple for teaching.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(length).mean()
    rs = gain / loss
    out = 100 - (100 / (1 + rs))
    return out

@dataclass
class RSIThresholdStrategy(Strategy):
    length: int = 14
    buy_below: float = 30.0
    sell_above: float = 70.0

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        r = rsi(df["close"], self.length)

        # Long when RSI crosses up through buy_below; exit long when it crosses down through sell_above
        long_entry = (r > self.buy_below) & (r.shift(1) <= self.buy_below)
        long_exit  = (r < self.sell_above) & (r.shift(1) >= self.sell_above)

        # Short when RSI crosses down through sell_above; exit short when it crosses up through buy_below
        short_entry = (r < self.sell_above) & (r.shift(1) >= self.sell_above)
        short_exit  = (r > self.buy_below) & (r.shift(1) <= self.buy_below)

        out["entry_long"]  = long_entry
        out["exit_long"]   = long_exit
        out["entry_short"] = short_entry
        out["exit_short"]  = short_exit
        return out

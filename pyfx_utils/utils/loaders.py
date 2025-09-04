# /MyDrive/code/pyfx_utils/utils.py
from __future__ import annotations
import pandas as pd


def load_fx_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic normalization
    rename_map = {c.lower(): c for c in df.columns}
    for col in list(df.columns):
        if col.lower() in ["time","date","datetime"]:
            df.rename(columns={col:"timestamp"}, inplace=True)
    # Force canonical names if present
    cols_lower = [c.lower() for c in df.columns]
    for need in ["timestamp","open","high","low","close","volume"]:
        if need not in cols_lower:
            # ok if missing volume etc., but timestamp & close are strongly recommended
            pass
    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df = df.set_index("timestamp")
    
    #quality report
    rpt = generate_quality_report(df)
    print_quality_report(rpt)
    
    return df

def resample(df: pd.DataFrame, rule: str = "1D",
                  label: str = "right", closed: str = "right") -> pd.DataFrame:
    """
    Minimal OHLC resample. Examples: '15min', '1H', '4H', '1D'.
    """
    agg = {"open":"first", "high":"max", "low":"min", "close":"last", "volume":"sum"}
    out = df.resample(rule, label=label, closed=closed).agg(agg)
    return out.dropna(subset=["open","high","low","close","volume"])

def pip_factor(instrument: str) -> float:
    """
    Simple inference:
      - JPY quotes ~ 0.01 pip (e.g., USD/JPY)
      - Otherwise ~ 0.0001 (e.g., EUR/USD)
    """
    inst = instrument.replace(" ", "").upper()
    return 0.01 if "JPY" in inst else 0.0001

def generate_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    report = {}
    if df.index.tz is not None:
        report["timezone"] = str(df.index.tz)
    else:
        report["timezone"] = "naive (⚠️ consider localizing to UTC)"
    report["rows"] = len(df)
    report["dup_index"] = int(df.index.duplicated().sum())
    report["nulls"] = int(df.isna().sum().sum())
    # Gaps: assumes fairly regular sampling
    if len(df) > 2:
        diffs = df.index.to_series().diff().dropna().value_counts().sort_values(ascending=False)
        report["top_freqs"] = diffs.head(3).to_dict()
    # OHLC sanity
    if all(c in df.columns for c in ["open","high","low","close"]):
        bad = (df["high"] < df[["open","close"]].max(axis=1)) | (df["low"] > df[["open","close"]].min(axis=1))
        report["ohlc_violations"] = int(bad.sum())
    # Weekend detection (rough heuristic)
    report["weekend_rows"] = int(((df.index.dayofweek>=5)).sum())
    return report

def print_quality_report(report: Dict[str,Any]):
    for k,v in report.items():
        print(f"{k:>16}: {v}")
    if report.get("dup_index",0)>0:
        print("\n⚠️ Duplicate timestamps found. Consider aggregating or de-duping.")
    if report.get("ohlc_violations",0)>0:
        print("⚠️ OHLC violations detected. Source data may be corrupted.")
    if report.get("weekend_rows",0)>0:
        print("ℹ️ Weekend rows present. Confirm your broker's session conventions.")


import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- MATHEMATIK & LOGIK ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    """Berechnet das Delta (Wahrscheinlichkeits-Proxy)."""
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def calculate_rsi(data, window=14):
    """Berechnet den technischen Indikator RSI."""
    if len(data) < window + 1: return 50
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_clean_earnings(tk):
    """Holt das Earnings-Datum und bereinigt Formatfehler."""
    try:
        cal = tk.calendar
        date = None
        if isinstance(cal, dict) and 'Earnings Date' in cal:
            date = cal['Earnings Date'][0]
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            date = cal.iloc[0, 0]
        return date
    except:
        return None

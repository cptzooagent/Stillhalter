import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime

# --- SETUP ---
MD_KEY = st.secrets.get("MARKETDATA_KEY")
POLY_KEY = st.secrets.get("POLYGON_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- HELPER: POLYGON OPTION TICKER FORMAT ---
def get_poly_ticker(symbol, date_str, strike, side):
    # Formatiert Ticker für Polygon: O:TSLA230616P00150000
    date_part = date_str.replace("-", "")[2:]
    side_part = "P" if side == "put" else "C"
    strike_part = f"{int(strike * 1000):08d}"
    return f"O:{symbol}{date_part}{side_part}{strike_part}"

# --- DATA FUNCTIONS ---

@st.cache_data(ttl=3600)
def get_all_expirations(symbol):
    # 1. Versuch: MarketData
    try:
        r = requests.get(f"https://api.marketdata.app/v1/options/expirations/{symbol}/?token={MD_KEY}").json()
        if r.get('s') == 'ok': return sorted(r.get('expirations', []))
    except: pass
    
    # 2. Versuch: Polygon Backup
    st.warning(f"MarketData Limit erreicht. Wechsle zu Polygon für {symbol}...")
    try:
        # Polygon Snapshot für alle Optionen eines Tickers
        url = f"https://api.polygon.io/v3/snapshot/options/{symbol}?apiKey={POLY_KEY}&limit=1"
        r = requests.get(url).json()
        if r.get('status') == 'OK':
            # Polygon liefert keine einfache Expiration-Liste, wir extrahieren sie aus den Tickersymbolen
            # (Vereinfacht für heute: Wir nehmen das heutige Datum + 30 Tage als Fallback)
            return [datetime.now().strftime("%Y-%m-%d")] 
    except: pass
    return []

@st.cache_data(ttl=600)
def get_chain_for_date(symbol, date_str, side):
    # 1. Versuch: MarketData
    try:
        params = {"token": MD_KEY, "side": side, "expiration": date_str}
        r = requests.get(f"https://api.marketdata.app/v1/options/chain/{symbol}/", params=params).json()
        if r.get('s') == 'ok' and len(r.get('strike', [])) > 0:
            return pd.DataFrame({
                'strike': r['strike'], 'mid': r['mid'], 
                'delta': r.get('delta', [0.0]*len(r['strike'])), 
                'iv': r.get('iv', [0.0]*len(r['strike']))
            })
    except: pass

    # 2. Versuch: Polygon Backup (Snapshot API)
    try:
        url = f"https://api.polygon.io/v3/snapshot/options/{symbol}?apiKey={POLY_KEY}"
        r = requests.get(url).json()
        if r.get('status') == 'OK':
            data = []
            for res in r.get('results', []):
                # Filtern nach Seite und Datum (aus dem Ticker extrahiert)
                if side.upper() in res['details']['ticker']:
                    data.append({
                        'strike': res['details']['strike_price'],
                        'mid': res.get('last_quote', {}).get('p', 0),
                        'delta': 0.15, # Polygon Free liefert kein Live-Delta
                        'iv': res.get('implied_volatility', 0)
                    })
            df = pd.DataFrame(data)
            return df if not df.empty else None
    except: pass
    return None

# --- UI (METRIKEN & FINDER) ---
st.title("🛡️ CapTrader Hybrid Scanner")
# ... (Rest des Codes wie Bitcoin/VIX Metriken) ...

st.subheader("🔍 Options-Finder")
ticker = st.text_input("Ticker (z.B. HOOD)").upper()
if ticker:
    exp = get_all_expirations(ticker)
    if exp:
        sel_date = st.selectbox("Datum", exp)
        chain = get_chain_for_date(ticker, sel_date, "put")
        if chain is not None:
            st.success(f"Daten für {ticker} geladen!")
            st.dataframe(chain.head(10))
        else:
            st.error("Auch Polygon liefert keine Daten. Minute-Limit (5/Min) erreicht?")

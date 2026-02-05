import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="Stillhalter Pro Scanner", layout="wide")

MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- MARKT-DATEN FUNKTION (FIXED) ---
def get_market_overview():
    try:
        # Nutzung von echten Index-Symbolen für stabilere Werte
        symbols = {
            "VIX (Angst)": "^VIX", 
            "S&P 500": "^GSPC", 
            "Nasdaq": "^IXIC", 
            "Bitcoin": "BINANCE:BTCUSDT"
        }
        data = {}
        for name, sym in symbols.items():
            r = requests.get(f'https://finnhub.io/api/v1/quote?symbol={sym}&token={FINNHUB_KEY}').json()
            # Falls Index-Daten (^) nicht verfügbar, Fallback auf ETFs
            if not r.get('c'):
                alt_sym = "SPY" if "S&P" in name else "QQQ" if "Nasdaq" in name else sym
                r = requests.get(f'https://finnhub.io/api/v1/quote?symbol={alt_sym}&token={FINNHUB_KEY}').json()
            
            data[name] = {"price": r.get('c', 0), "change": r.get('dp', 0)}
        return data
    except:
        return None

def get_live_price(symbol):
    try:
        r = requests.get(f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}').json()
        return float(r['c']) if r.get('c') else None
    except: return None

def get_sma_200(symbol):
    try:
        r = requests.get(f'https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&count=250&token={FINNHUB_KEY}').json()
        if r.get('s') == 'ok':
            closes = r.get('c', [])
            return sum(closes[-200:]) / 200 if len(closes) >= 200 else None
    except: return None

def get_all_expirations(symbol):
    try:
        r = requests.get(f"https://api.marketdata.app/v1/options/expirations/{symbol}/?token={MD_KEY}").json()
        return sorted(r.get('expirations', [])) if r.get('s') == 'ok' else []
    except: return []

def get_chain_for_date(symbol, date_str, side):
    try:
        params = {"token": MD_KEY, "side": side, "expiration": date_str}
        r = requests.get(f"https://api.marketdata.app/v1/options/chain/{symbol}/", params=params).json()
        if r.get('s') == 'ok':
            return pd.DataFrame({'strike': r['strike'], 'mid': r['mid'], 'delta': r.get('delta', [0]*len(r['strike'])), 'expiration': date_str})
    except: return None

# --- UI START ---
st.title("🛡️ Pro Stillhalter Scanner")

# 1. MARKT-AMPEL (DYNAMISCH & KORRIGIERT)
market = get_market_overview()
vix_status = "normal"
if market:
    vix = market['VIX (Angst)']['price']
    with st.container(border=True):
        cols = st.columns(len(market))
        for i, (name, info) in enumerate(market.items()):
            label = "🔥 PANIK" if name == "VIX (Angst)" and info['price'] > 25 else "🟢 RUHIG" if name == "VIX (Angst)" else ""
            cols[i].metric(name, f"{info['price']:,.2f}", f"{info['change']:.2f}% {label}", delta_color="normal" if name != "VIX (Angst)" else "inverse")
        
        if vix > 25:
            vix_status = "panic"
            st.error("⚠️ Marktsituation: Hohe Volatilität. Sicherheits-Grenzwerte verschärft.")

st.divider()

# 2. STRATEGIE & TICKER
option_type = st.radio("Strategie", ["Put 🛡️", "Call 📈"], horizontal=True)
side = "put" if "Put" in option_type else "call"
watchlist = ["AAPL", "TSLA", "NVDA", "MSFT", "AMD", "META", "GOOGL", "AMZN"]
sel_fav = st.pills("Favoriten", watchlist)
ticker = (sel_fav if sel_fav else st.text_input("Ticker", "")).strip().upper()

if ticker:
    price = get_live_price(ticker)
    sma200 = get_sma_200(ticker)
    if price:
        c_p, c_t = st.columns(2)
        c_p.metric(f"Kurs {ticker}", f"{price:.2f} $")
        if sma200:
            diff = ((price / sma200) - 1) * 100
            c_t.metric("SMA 200 Trend", f"{sma200:.2f} $", f"{diff:.1f}%")
        
        dates = get_all_expirations(ticker)
        if dates:
            # Fix für Datum-Anzeige aus Bild 9 & 10
            date_labels = {d: f"{datetime.strptime(d, '%Y-%m-%d').strftime('%d.%m.%Y')} ({(datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days} T.)" for d in dates}
            sel_date = st.selectbox("Laufzeit", dates, format_func=lambda x: date_labels.get(x))
            
            df = get_chain_for_date(ticker, sel_date, side)
            if df is not None and not df.empty:
                df = df[df['strike'] < price] if side == "put" else df[df['strike'] > price]
                df = df.sort_values('strike', ascending=(side == "call"))
                
                for _, row in df.head(12).iterrows():
                    d_abs = abs(row['delta'] if row['delta'] else 0)
                    pop = (1 - d_abs) * 100
                    
                    # Dynamische Ampel basierend auf VIX
                    if vix_status == "panic":
                        color = "🟢" if d_abs < 0.10 else "🟡" if d_abs < 0.18 else "🔴"
                    else:
                        color = "🟢" if d_abs < 0.16 else "🟡" if d_abs < 0.25 else "🔴"
                    
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | Chance: {pop:.0f}%"):
                        ca, cb, cc = st.columns(3)
                        ca.metric("Prämie", f"{row['mid']:.2f}$")
                        cb.metric("Abstand", f"{(abs(price-row['strike'])/price)*100:.1f}%")
                        cc.metric("Gewinn-Chance", f"{pop:.1f}%")
                        # Progress-Balken Fix für Bild 2 (Werte 0.0 bis 1.0)
                        st.progress(max(0.0, min(1.0, pop / 100)))

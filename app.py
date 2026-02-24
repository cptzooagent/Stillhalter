import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime
import time
import requests

# --- 1. SESSION FIX (GEGEN TOO MANY REQUESTS) ---
def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session

# --- 2. MATHEMATIK & INDIKATOREN ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def calculate_rsi(data, window=14):
    if len(data) < window + 1: return pd.Series([50] * len(data))
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 3. MARKT-DATEN FUNKTIONEN ---
def get_market_metrics(session):
    try:
        vix = yf.Ticker("^VIX", session=session).history(period="1d")['Close'].iloc[-1]
        ndq = yf.Ticker("^NDX", session=session).history(period="30d")
        cp_ndq = ndq['Close'].iloc[-1]
        sma20_ndq = ndq['Close'].rolling(window=20).mean().iloc[-1]
        dist_ndq = ((cp_ndq - sma20_ndq) / sma20_ndq) * 100
        return cp_ndq, dist_ndq, vix
    except:
        return 0, 0, 20

# --- 4. SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")
session = get_session()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛡️ Scanner-Filter")
    otm_puffer_val = st.slider("OTM Puffer (%)", 5, 25, 15)
    min_yield_pa = st.number_input("Mindestrendite p.a. (%)", 5, 100, 12)
    st.markdown("---")
    st.info("Scanner pausiert 1 Sek. pro Ticker, um die API-Sperre zu umgehen.")

# --- 5. GLOBALER MONITOR ---
st.title("🌍 Globales Markt-Monitoring")
cp_ndq, dist_ndq, vix_val = get_market_metrics(session)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Nasdaq 100", f"{cp_ndq:,.0f}", f"{dist_ndq:.2f}% vs SMA20")
with col2:
    st.metric("VIX (Angst-Index)", f"{vix_val:.2f}", 
              delta="Nervös" if vix_val > 22 else "Ruhig", 
              delta_color="inverse")
with col3:
    # Marktstimmung basierend auf Volatilität
    if vix_val > 25: status, color = "Panik (Puts teuer!)", "red"
    elif vix_val > 20: status, color = "Erhöht (Gute Prämien)", "orange"
    else: status, color = "Ruhig (Wenig Prämie)", "green"
    st.markdown(f"Status: **:{color}[{status}]**")

st.markdown("---")

# --- 6. PROFI-SCANNER ---
st.header("🎯 Profi-Optionen Scanner")

if st.button("🚀 Scan starten"):
    # Ticker-Liste für den Test (kann beliebig erweitert werden)
    ticker_liste = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "PLTR", "COIN", "MSTR", "NFLX"]
    
    results = []
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(ticker_liste):
        status_text.text(f"Analysiere {symbol} ({i+1}/{len(ticker_liste)})...")
        try:
            tk = yf.Ticker(symbol, session=session)
            hist = tk.history(period="150d")
            
            if hist.empty: continue
            
            price = hist['Close'].iloc[-1]
            rsi = calculate_rsi(hist['Close']).iloc[-1]
            sma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            
            # Optionen laden
            opt_dates = tk.options
            if not opt_dates: continue
            
            # Wir nehmen das erste verfügbare Datum (nächster Verfall)
            target_date = opt_dates[0]
            chain = tk.option_chain(target_date).puts
            
            # Filter nach OTM-Puffer
            target_strike = price * (1 - (otm_puffer_val / 100))
            valid_puts = chain[chain['strike'] <= target_strike].sort_values('strike', ascending=False)
            
            if not valid_puts.empty:
                put = valid_puts.iloc[0]
                # Preis-Findung (Bid oder Last)
                bid = put['bid'] if put['bid'] > 0 else put['lastPrice']
                
                # Berechnung Tage bis Verfall
                days_to_exp = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days
                days_to_exp = max(1, days_to_exp)
                
                # Rendite p.a.
                y_pa = (bid / put['strike']) * (365 / days_to_exp) * 100
                
                if y_pa >= min_yield_pa:
                    results.append({
                        "Symbol": symbol,
                        "Aktienkurs": round(price, 2),
                        "Strike": put['strike'],
                        "OTM Puffer": f"{round(((price-put['strike'])/price)*100, 1)}%",
                        "Rendite p.a.": f"{round(y_pa, 1)}%",
                        "Delta": round(calculate_bsm_delta(price, put['strike'], days_to_exp/365, 0.4), 2),
                        "RSI (14)": int(rsi),
                        "Trend (SMA200)": "🟢" if price > sma200 else "🔴"
                    })
            
            # Sicherheits-Pause gegen API-Sperre
            time.sleep(1.0)
            
        except Exception:
            continue
            
        progress_bar.progress((i + 1) / len(ticker_liste))

    status_text.empty()
    progress_bar.empty()

    if results:
        st.subheader(f"💎 Gefundene Setups ({len(results)})")
        df_res = pd.DataFrame(results)
        # Tabelle schön formatieren
        st.dataframe(df_res.style.background_gradient(subset=['RSI (14)'], cmap='RdYlGn_r'), use_container_width=True)
    else:
        st.warning("Keine Setups gefunden. Versuche den OTM-Puffer zu senken oder die Rendite-Anforderung anzupassen.")

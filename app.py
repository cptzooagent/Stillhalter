import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime
import time
import requests

# --- 1. SESSION & BROWSER SIMULATION ---
def get_session():
    """Erzeugt eine Session, die Yahoo Finance einen echten Browser vorgaukelt."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session

# --- 2. OPTIONEN-MATHEMATIK ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    """Berechnet das Delta für Optionen (Black-Scholes)."""
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def calculate_rsi(data, window=14):
    """Berechnet den Relative Strength Index."""
    if len(data) < window + 1: return pd.Series([50] * len(data))
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 3. MARKT-DATEN ENGINE ---
def get_market_metrics(session):
    """Holt globale Daten für Nasdaq und Volatilität."""
    try:
        vix = yf.Ticker("^VIX", session=session).history(period="1d")['Close'].iloc[-1]
        ndq = yf.Ticker("^NDX", session=session).history(period="30d")
        cp_ndq = ndq['Close'].iloc[-1]
        sma20_ndq = ndq['Close'].rolling(window=20).mean().iloc[-1]
        dist_ndq = ((cp_ndq - sma20_ndq) / sma20_ndq) * 100
        return cp_ndq, dist_ndq, vix
    except:
        return 0, 0, 20

# --- 4. STREAMLIT UI SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")
session = get_session()

with st.sidebar:
    st.header("🛡️ Scanner-Strategie")
    otm_puffer = st.slider("Gewünschter Puffer OTM (%)", 5, 25, 15)
    min_yield = st.number_input("Mindestrendite p.a. (%)", 5, 100, 12)
    st.markdown("---")
    ticker_input = st.text_area("Ticker-Liste (getrennt durch Komma)", 
                                "AAPL,MSFT,NVDA,AMD,TSLA,GOOGL,AMZN,META,COIN,MSTR,PLTR")
    st.caption("Lokaler Modus: Pausiert 1s pro Aktie für Stabilität.")

# --- 5. GLOBALER MONITOR ---
st.title("🌍 Globales Markt-Monitoring")
cp_ndq, dist_ndq, vix_val = get_market_metrics(session)

r1, r2, r3 = st.columns(3)
with r1:
    st.metric("Nasdaq 100", f"{cp_ndq:,.0f}", f"{dist_ndq:.2f}% vs SMA20")
with r2:
    st.metric("VIX (Fear Index)", f"{vix_val:.2f}", delta="Gefahr" if vix_val > 22 else "Ruhig", delta_color="inverse")
with r3:
    status = "🔴 Panik" if vix_val > 25 else "🟡 Nervös" if vix_val > 20 else "🟢 Entspannt"
    st.write(f"Markt-Status: **{status}**")
    st.progress(min(vix_val / 50, 1.0))

st.markdown("---")

# --- 6. DER PROFI-SCANNER ---
st.header("🎯 Profi-Optionen Scanner")

if st.button("🚀 Profi-Scan starten"):
    ticker_list = [t.strip().upper() for t in ticker_input.split(",")]
    results = []
    
    status_box = st.empty()
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(ticker_list):
        status_box.info(f"Analysiere {symbol} ({i+1}/{len(ticker_list)})...")
        try:
            tk = yf.Ticker(symbol, session=session)
            hist = tk.history(period="200d")
            
            if hist.empty: continue
            
            # Basiswerte
            price = hist['Close'].iloc[-1]
            rsi = calculate_rsi(hist['Close']).iloc[-1]
            sma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            uptrend = price > sma200
            
            # Optionsdaten abrufen
            opt_dates = tk.options
            if not opt_dates: continue
            
            # Ziel: Nächster Monatsverfall (vereinfacht: erstes Datum)
            target_date = opt_dates[0]
            chain = tk.option_chain(target_date).puts
            
            # Strike-Ermittlung basierend auf OTM-Puffer
            target_strike = price * (1 - (otm_puffer / 100))
            valid_puts = chain[chain['strike'] <= target_strike].sort_values('strike', ascending=False)
            
            if not valid_puts.empty:
                put = valid_puts.iloc[0]
                bid = put['bid'] if put['bid'] > 0 else put['lastPrice']
                
                # Laufzeit berechnen
                days_to_exp = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days
                days_to_exp = max(1, days_to_exp)
                
                # Rendite p.a. Kalkulation
                # Formel: (Prämie / Strike) * (365 / Tage)
                y_pa = (bid / put['strike']) * (365 / days_to_exp) * 100
                
                if y_pa >= min_yield:
                    results.append({
                        "Symbol": symbol,
                        "Kurs": round(price, 2),
                        "Strike": put['strike'],
                        "Abstand": f"{round(((price-put['strike'])/price)*100, 1)}%",
                        "Rendite p.a.": f"{round(y_pa, 1)}%",
                        "Delta": round(calculate_bsm_delta(price, put['strike'], days_to_exp/365, 0.4), 2),
                        "RSI (14)": int(rsi),
                        "Trend": "🟢 Bullish" if uptrend else "🔴 Bearish"
                    })
            
            # Anti-Sperr-Pause
            time.sleep(1.2)
            
        except Exception as e:
            st.error(f"Fehler bei {symbol}: {str(e)}")
            continue
            
        progress_bar.progress((i + 1) / len(ticker_list))

    status_box.empty()
    progress_bar.empty()

    if results:
        st.subheader(f"✅ Top-Setups nach deinen Filtern ({len(results)})")
        df = pd.DataFrame(results)
        
        # Design der Tabelle
        st.dataframe(df.style.background_gradient(subset=['RSI (14)'], cmap='RdYlGn_r'), 
                     use_container_width=True)
    else:
        st.warning("Keine Setups gefunden. Verringere evtl. den Puffer oder die Mindestrendite.")

# --- 7. DEPOT-HINWEIS ---
st.markdown("---")
st.caption("Datenquelle: Yahoo Finance | Analyse für CapTrader Cash-Secured Puts")

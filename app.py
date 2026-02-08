import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- 1. CORE MATH (STABIL) ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    try:
        if T <= 0 or sigma <= 0 or S <= 0: return 0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
        return float(delta)
    except:
        return 0

# --- 2. DATA ENGINE (VERHINDERT DAS ZERSCHIESSEN) ---
@st.cache_data(ttl=600)
def fetch_safe_data(ticker_symbol):
    try:
        tk = yf.Ticker(ticker_symbol)
        # Schneller Preis-Check
        h = tk.history(period="5d")
        if h.empty: return None
        
        current_price = h['Close'].iloc[-1]
        
        # RSI Berechnung
        delta = h['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=3).mean() # Kurzzeit-RSI für Stabilität
        loss = (-delta.where(delta < 0, 0)).rolling(window=3).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        
        return {
            "price": current_price,
            "rsi": rsi,
            "options_dates": tk.options,
            "ticker_obj": tk
        }
    except:
        return None

# --- UI SETUP ---
st.set_page_config(page_title="CapTrader AI Scanner", layout="wide")
st.title("🛡️ CapTrader AI Market Scanner")

# SEKTION: DEPOT PERFORMANCE (STATISCH ODER DYNAMISCH)
st.subheader("📊 Meine Depot-Performance")
c1, c2, c3 = st.columns(3)
c1.metric("Depotwert", "42.500 $", "1.2%")
c2.metric("Verfügbare Margin", "28.400 $", "-0.5%")
c3.metric("Monatlicher Cashflow", "1.150 $", "8%")

st.divider()

# SEKTION: EINZEL-CHECK
st.subheader("🔍 Deep-Dive Einzel-Check")

col_left, col_right = st.columns([1, 3])
with col_left:
    opt_type = st.radio("Typ", ["put", "call"], horizontal=True)
with col_right:
    ticker_input = st.text_input("Symbol eingeben", value="HOOD").upper()

if ticker_input:
    data = fetch_safe_data(ticker_input)
    
    if data:
        st.write(f"Aktueller Kurs: **{data['price']:.2f}$** | RSI: **{data['rsi']:.0f}**")
        
        if data['options_dates']:
            selected_date = st.selectbox("Laufzeit wählen", data['options_dates'])
            
            # Optionsdaten laden
            try:
                chain = data['ticker_obj'].option_chain(selected_date)
                opts = chain.puts if opt_type == "put" else chain.calls
                
                T = (datetime.strptime(selected_date, '%Y-%m-%d') - datetime.now()).days / 365
                
                # Delta für alle Optionen berechnen
                opts['delta'] = opts.apply(lambda x: calculate_bsm_delta(
                    data['price'], x['strike'], T, x['impliedVolatility'] or 0.4, option_type=opt_type
                ), axis=1)

                # Anzeige der Top-Optionen
                st.write("---")
                for _, opt in opts.sort_values('strike', ascending=(opt_type=="call")).head(5).iterrows():
                    d_abs = abs(opt['delta'])
                    with st.expander(f"Strike {opt['strike']:.1f}$ | Delta: {d_abs:.2f} | Bid: {opt['bid']:.2f}$"):
                        st.write(f"Wahrscheinlichkeit OTM: **{(1-d_abs)*100:.1f}%**")
                        st.write(f"Implizite Volatilität: **{opt['impliedVolatility']*100:.1f}%**")
            except Exception as e:
                st.error(f"Konnte Optionskette nicht laden: {e}")
    else:
        st.error("Ticker konnte nicht gefunden werden. Bitte Symbol prüfen.")

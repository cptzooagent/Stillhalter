import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader Pro", layout="wide")

# Stabilisierte Mathe-Engine
def get_delta(S, K, T, sigma, cp='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (0.04 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1 if cp == 'put' else norm.cdf(d1)

def get_rsi(data, window=14):
    if len(data) < window + 1: return 50
    delta = data.diff()
    up = delta.clip(lower=0).rolling(window=window).mean()
    down = -delta.clip(upper=0).rolling(window=window).mean()
    return 100 - (100 / (1 + up/down))

# FIX für Bild 11: Wir cachen nur primitive Datentypen (Keine Ticker-Objekte!)
@st.cache_data(ttl=600)
def get_clean_data(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        hist = tk.history(period="1mo")
        rsi_val = get_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        opts = list(tk.options)
        earn = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return price, opts, rsi_val, earn
    except:
        return None, [], 50, ""

# --- UI SIDEBAR ---
st.sidebar.header("Strategie")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
min_yield = st.sidebar.number_input("Min. Rendite p.a. (%)", value=15)
st.sidebar.markdown("---")
st.sidebar.info("💡 RSI immer mit Support-Zonen im Chart abgleichen!")

st.title("🛡️ CapTrader AI Intelligence")

# --- SCANNER ---
if st.button("🚀 Markt-Analyse starten", use_container_width=True):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "PLTR", "HOOD", "AFRM"]
    results = []
    
    for t in watchlist:
        p, dates, rsi, earn = get_clean_data(t)
        if p and dates:
            try:
                tk = yf.Ticker(t) # Ticker lokal ohne Cache laden
                d_target = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(d_target).puts
                T = (datetime.strptime(d_target, '%Y-%m-%d') - datetime.now()).days / 365
                
                chain['delta'] = chain.apply(lambda r: get_delta(p, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                matches = chain[chain['delta'].abs() <= (1 - target_prob/100)].copy()
                
                if not matches.empty:
                    matches['y_pa'] = (matches['bid'] / matches['strike']) * (1/T) * 100
                    best = matches.sort_values('y_pa', ascending=False).iloc[0]
                    if best['y_pa'] >= min_yield:
                        results.append({'T': t, 'Y': best['y_pa'], 'S': best['strike'], 'B': best['bid'], 'D': abs(best['delta']), 'R': rsi})
            except: continue

    if results:
        cols = st.columns(3)
        for i, r in enumerate(results):
            with cols[i % 3]:
                st.metric(r['T'], f"{r['Y']:.1f}% p.a.", f"Δ {r['D']:.2f}")
                with st.expander(f"Details: {r['S']}$ Strike"):
                    st.write(f"💵 Prämie: **{r['B']:.2f}$**")
                    st.write(f"📊 RSI: **{r['R']:.0f}**")
    else: st.warning("Keine Treffer.")

# --- EINZEL-CHECK (DELTA GARANTIERT) ---
st.markdown("---")
st.subheader("🔍 Einzel-Check")
ticker_in = st.text_input("Symbol", "NVDA").upper()
if ticker_in:
    p, dates, rsi, earn = get_clean_data(ticker_in)
    if p:
        st.write(f"Kurs: **{p:.2f}$** | RSI: **{rsi:.0f}**")
        sel_date = st.selectbox("Laufzeit", dates)
        tk = yf.Ticker(ticker_in)
        chain = tk.option_chain(sel_date).puts
        T = (datetime.strptime(sel_date, '%Y-%m-%d') - datetime.now()).days / 365
        
        chain['delta'] = chain.apply(lambda r: get_delta(p, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
        
        # Wichtig: Delta und Prämie fest im Titel
        for _, row in chain[chain['delta'].abs() < 0.35].sort_values('strike', ascending=False).head(5).iterrows():
            with st.expander(f"Strike {row['strike']}$ | Δ {abs(row['delta']):.2f} | Bid: {row['bid']:.2f}$"):
                st.write(f"💰 Cash-Einnahme: **{row['bid']*100:.0f}$**")
                st.write(f"📉 Puffer: **{abs(row['strike']-p)/p*100:.1f}%**")

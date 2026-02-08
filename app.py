import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader Pro Scanner", layout="wide")

# --- 1. MATHE-KERN ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    except:
        return 0

def calculate_rsi(data, window=14):
    if len(data) < window + 1: return 50
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# --- 2. DATEN-ABRUF ---
@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        hist = tk.history(period="1mo")
        rsi_val = calculate_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
        except: earn_str = ""
        
        return price, list(tk.options), earn_str, rsi_val
    except:
        return None, [], "", 50

# --- UI ---
st.title("🛡️ CapTrader AI Market Scanner")

# --- SEKTION: EINZEL-CHECK (Fix für falsche Strikes) ---
st.write("---")
st.subheader("🔍 Deep-Dive Einzel-Check")

c_type, c_tick = st.columns([1, 3])
opt_type = c_type.radio("Strategie wählen", ["put", "call"], horizontal=True)
t_in = c_tick.text_input("Aktien-Symbol (z.B. HOOD, NVDA, TSLA)", "HOOD").upper()

if t_in:
    price, dates, earn, rsi = get_stock_data_full(t_in)
    
    if price and dates:
        st.write(f"Aktueller Kurs: **{price:.2f}$** | RSI: **{rsi:.0f}**")
        expiry = st.selectbox("Laufzeit wählen", dates)
        
        tk = yf.Ticker(t_in)
        chain = tk.option_chain(expiry).calls if opt_type == "call" else tk.option_chain(expiry).puts
        T = max(1/365, (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days / 365)
        
        # --- FILTER-LOGIK ---
        if opt_type == "put":
            # Wir suchen Puts UNTER dem aktuellen Kurs (OTM)
            display_chain = chain[chain['strike'] < price].sort_values('strike', ascending=False)
        else:
            # Wir suchen Calls ÜBER dem aktuellen Kurs (OTM)
            display_chain = chain[chain['strike'] > price].sort_values('strike', ascending=True)
        
        # Nur die nächsten 8 relevanten Strikes anzeigen
        display_chain = display_chain.head(8)
        
        if display_chain.empty:
            st.warning("Keine passenden OTM-Optionen für diese Laufzeit gefunden.")
        else:
            for _, opt in display_chain.iterrows():
                # Delta für jeden Strike berechnen
                delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, option_type=opt_type)
                d_abs = abs(delta)
                
                # Risiko-Einstufung
                if d_abs < 0.16: label, color = "(Sicher)", "🟢"
                elif d_abs < 0.31: label, color = "(Moderat)", "🟡"
                else: label, color = "(Aggressiv)", "🔴"
                
                with st.expander(f"{color} {label} Strike {opt['strike']:.1f}$ | Delta: {d_abs:.2f}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"💰 **Prämie:** {opt['bid']*100:.0f}$ (pro Kontrakt)")
                        st.write(f"📉 **Abstand:** {abs(opt['strike']-price)/price*100:.1f}% vom Kurs")
                    with col2:
                        st.write(f"🎯 **OTM-Chance:** {int((1-d_abs)*100)}%")
                        st.write(f"🌊 **Impl. Vola:** {int((opt['impliedVolatility'] or 0)*100)}%")

# --- SEKTION: MARKT-SCAN ---
st.write("---")
if st.button("🚀 Markt-Scan (Top 12 Watchlist)", use_container_width=True):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "PLTR", "HOOD", "AFRM", "AMD", "COIN"]
    res_cols = st.columns(3)
    idx = 0
    
    for t in watchlist:
        p, dts, e, r = get_stock_data_full(t)
        if p and dts:
            try:
                target_date = min(dts, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                tk_obj = yf.Ticker(t)
                puts = tk_obj.option_chain(target_date).puts
                T_scan = max(1/365, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365)
                
                # Finde sichersten Put (Delta ~ 0.15)
                puts['d'] = puts.apply(lambda row: abs(calculate_bsm_delta(p, row['strike'], T_scan, row['impliedVolatility'] or 0.4)), axis=1)
                best_put = puts[puts['d'] <= 0.20].sort_values('bid', ascending=False).iloc[0]
                
                with res_cols[idx % 3]:
                    st.info(f"**{t}** | RSI: {r:.0f}")
                    st.write(f"Strike: {best_put['strike']}$ | Delta: {best_put['d']:.2f}")
                    st.write(f"Prämie: {best_put['bid']*100:.0f}$")
                idx += 1
            except: continue

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 1. MATHE-LOGIK ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def calculate_rsi(data, window=14):
    if len(data) < window + 1: return 50
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 2. DATEN-ABRUF ---
@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        hist = tk.history(period="1mo")
        rsi_val = calculate_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        
        earn_str = ""
        earn_dt = None
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_dt = cal['Earnings Date'][0]
                earn_str = earn_dt.strftime('%d.%m.')
        except: pass
        
        return price, dates, earn_str, rsi_val, earn_dt
    except:
        return None, [], "", 50, None

# --- SIDEBAR ---
st.sidebar.header("🛡️ Strategie-Einstellungen")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 83)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)

st.title("🛡️ CapTrader AI Market Scanner")

# --- SEKTION 1: MARKT-SCAN ---
if st.button("🚀 Kombi-Scan starten"):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "PLTR", "HOOD", "AFRM", "COIN", "MSTR"]
    results = []
    prog = st.progress(0)
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        price, dates, earn, rsi, _ = get_stock_data_full(t)
        if price and dates:
            try:
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                tk = yf.Ticker(t)
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                chain['delta'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                matches = chain[chain['delta'].abs() <= max_delta]
                if not matches.empty:
                    best = matches.sort_values('strike', ascending=False).iloc[0]
                    y_pa = (best['bid'] / best['strike']) * (365 / max(1, T*365)) * 100
                    if y_pa >= min_yield_pa:
                        results.append({'Ticker': t, 'Rendite': f"{y_pa:.1f}%", 'Strike': best['strike'], 'Delta': f"{abs(best['delta']):.2f}", 'RSI': int(rsi)})
            except: continue
    if results: st.table(pd.DataFrame(results))

st.divider()

# --- SEKTION 2: DEPOT ---
st.subheader("💼 Smart Depot-Manager")
depot = [{"T": "AFRM", "E": 76.0}, {"T": "HOOD", "E": 82.82}, {"T": "PLTR", "E": 25.0}, {"T": "MSTR", "E": 1500.0}]
d_cols = st.columns(len(depot))
for idx, item in enumerate(depot):
    price, _, earn, rsi, earn_dt = get_stock_data_full(item['T'])
    if price:
        diff = (price/item['E']-1)*100
        with d_cols[idx]:
            st.metric(item['T'], f"{price:.2f}$", f"{diff:.1f}%")
            if rsi > 65: st.success("Call?")
            if rsi < 35: st.info("Hold")

st.divider()

# --- SEKTION 3: EINZEL-CHECK (FIXED) ---
st.subheader("🔍 Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: t_in = st.text_input("Ticker", value="HOOD").upper()

if t_in:
    price, dates, earn, rsi, _ = get_stock_data_full(t_in)
    if price and dates:
        st.write(f"Kurs: **{price:.2f}$** | RSI: **{rsi:.0f}** | ER: {earn}")
        d_sel = st.selectbox("Laufzeit", dates)
        try:
            tk = yf.Ticker(t_in)
            chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
            T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
            chain['delta_calc'] = chain.apply(lambda o: calculate_bsm_delta(price, o['strike'], T, o['impliedVolatility'] or 0.4, mode), axis=1)
            
            # Anzeige-Filter
            df = chain[chain['strike'] <= price * 1.1] if mode == "put" else chain[chain['strike'] >= price * 0.9]
            for _, opt in df.sort_values('strike', ascending=(mode == "call")).head(15).iterrows():
                d_abs = abs(opt['delta_calc'])
                color = "🟢" if d_abs < 0.16 else "🟡" if d_abs <= 0.30 else "🔴"
                y_pa = (opt['bid'] / opt['strike']) * (365 / max(1, T*365)) * 100
                puffer = (abs(opt['strike']-price)/price)*100
                
                # DER FIX: Sauber geschlossene f-strings
                st.markdown(
                    f"{color} **Strike: {opt['strike']:.1f}** | "
                    f"Bid: <span style='color:#2ecc71;'>{opt['bid']:.2f}$</span> | "
                    f"Delta: {d_abs:.2f} | Puffer: {puffer:.1f}% | Rendite: {y_pa:.1f}% p.a.",
                    unsafe_allow_html=True
                )
        except Exception as e: st.error(f"Fehler: {e}")

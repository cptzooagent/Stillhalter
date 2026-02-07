import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- 1. SETUP & STABLE STYLING ---
st.set_page_config(page_title="CapTrader AI Scanner", layout="wide")

# CSS für Karten-Optik ohne Absturzgefahr
st.markdown("""
    <style>
    .stMetric { border: 1px solid #d1d5db; padding: 10px; border-radius: 5px; background: white; }
    .main-title { font-size: 2rem; font-weight: bold; color: #1E3A8A; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MATHE & DATEN ---
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

@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        hist = tk.history(period="1mo")
        rsi_val = calculate_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return price, dates, earn_str, rsi_val
    except:
        return None, [], "", 50

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("Strategie")
    target_prob = st.slider("Sicherheit (OTM %)", 70, 98, 85)
    max_delta = (100 - target_prob) / 100
    min_yield_pa = st.number_input("Min. Rendite p.a. (%)", value=15)
    sort_rsi = st.checkbox("Nach RSI sortieren")

st.markdown('<p class="main-title">🛡️ CapTrader AI Scanner</p>', unsafe_allow_html=True)

# --- 4. SCANNER (STABIL) ---
if st.button("🚀 Markt-Analyse starten", use_container_width=True):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "PLTR", "HOOD"]
    results = []
    prog = st.progress(0)
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        price, dates, earn, rsi = get_stock_data_full(t)
        if price and dates:
            try:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                safe_opts = chain[chain['delta_val'].abs() <= max_delta].copy()
                if not safe_opts.empty:
                    days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                    safe_opts['y_pa'] = (safe_opts['bid'] / safe_opts['strike']) * (365 / days) * 100
                    matches = safe_opts[safe_opts['y_pa'] >= min_yield_pa]
                    if not matches.empty:
                        best = matches.sort_values('y_pa', ascending=False).iloc[0]
                        results.append({'ticker': t, 'yield': best['y_pa'], 'strike': best['strike'], 'bid': best['bid'], 
                                        'puffer': (abs(best['strike'] - price) / price) * 100, 'delta': abs(best['delta_val']), 'rsi': rsi})
            except: continue

    if results:
        df_res = pd.DataFrame(results)
        opp_df = df_res.sort_values('rsi' if sort_rsi else 'yield', ascending=False)
        cols = st.columns(3)
        for idx, row in enumerate(opp_df.to_dict('records')):
            with cols[idx % 3]:
                st.metric(f"{row['ticker']}", f"{row['yield']:.1f}% p.a.", f"Δ {row['delta']:.2f}")
                st.write(f"**Strike:** {row['strike']:.1f}$ | **Bid:** {row['bid']:.2f}$")

# --- 5. DEPOT-MANAGER (FIX) ---
st.markdown("---")
st.subheader("💼 Depot-Manager")
depot = [{"T": "AFRM", "E": 76.0}, {"T": "HOOD", "E": 82.8}, {"T": "NVDA", "E": 115.0}]
d_cols = st.columns(3)
for i, item in enumerate(depot):
    p, _, earn, rsi = get_stock_data_full(item['T'])
    if p:
        perf = (p / item['E'] - 1) * 100
        with d_cols[i % 3]:
            with st.expander(f"{item['T']} ({perf:+.1f}%)", expanded=True):
                st.write(f"Kurs: **{p:.2f}$** | RSI: **{rsi:.0f}**")
                if rsi > 65: st.success("🎯 Call-Verkauf prüfen")
                if earn: st.caption(f"Earnings: {earn}")

# --- 6. EINZEL-CHECK (DELTA & PRÄMIE GARANTIERT) ---
st.markdown("---")
st.subheader("🔍 Einzel-Check")
ec1, ec2 = st.columns([1, 2])
with ec1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with ec2: t_in = st.text_input("Symbol", value="NVDA").upper()

if t_in:
    price, dates, earn, rsi = get_stock_data_full(t_in)
    if price and dates:
        st.write(f"Kurs: **{price:.2f}$** | RSI: **{rsi:.0f}**")
        d_sel = st.selectbox("Laufzeit", dates)
        try:
            tk = yf.Ticker(t_in)
            chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
            T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
            chain['delta_calc'] = chain.apply(lambda opt: calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, mode), axis=1)
            
            for _, opt in chain[chain['delta_calc'].abs() <= 0.4].sort_values('strike', ascending=(mode=="call")).head(5).iterrows():
                d_abs = abs(opt['delta_calc'])
                # HIER: Delta und Prämie fest in der Überschrift
                with st.expander(f"Strike {opt['strike']:.1f}$ | Delta: {d_abs:.2f} | Bid: {opt['bid']:.2f}$"):
                    c1, c2, c3 = st.columns(3)
                    c1.write(f"💰 **Cash:**\n{opt['bid']*100:.0f}$")
                    c2.write(f"🎯 **Puffer:**\n{(abs(opt['strike']-price)/price)*100:.1f}%")
                    c3.write(f"🌊 **IV:**\n{int((opt['impliedVolatility'] or 0)*100)}%")
        except: st.error("Fehler beim Laden.")

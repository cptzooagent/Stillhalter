import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Guard", layout="wide")

# --- 1. MATHE: DELTA-BERECHNUNG ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    T = max(T, 0.0001) 
    sigma = max(sigma, 0.0001)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

# --- 2. SICHERHEITS-LOGIK (Kein Caching von Objekten) ---
def get_safety_metrics(symbol):
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period="3mo")
        if len(hist) < 20: return 50.0, True
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        sma50 = hist['Close'].tail(50).mean()
        trend_ok = hist['Close'].iloc[-1] > (sma50 * 0.95)
        return round(float(rsi), 1), bool(trend_ok)
    except: return 50.0, True

@st.cache_data(ttl=900)
def get_stock_data(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        earn = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return price, dates, earn
    except: return None, [], ""

# --- 3. UI: SIDEBAR ---
st.sidebar.header("🛡️ Strategie & Sicherheit")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)
rsi_min = st.sidebar.slider("Minimum RSI", 20, 45, 30)

# --- 4. HAUPTBEREICH: SCANNER ---
st.title("🛡️ CapTrader AI Market Guard Pro")

if st.button("🚀 High-Safety Scan starten"):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "AMD", "NFLX", "COIN", "PLTR", "HOOD", "SQ", "MSTR", "SMCI", "UBER", "ABNB", "DIS", "PYPL"]
    results = []
    prog = st.progress(0)
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        price, dates, earn = get_stock_data(t)
        if price and dates:
            rsi, trend_ok = get_safety_metrics(t)
            if rsi >= rsi_min and trend_ok:
                try:
                    tk = yf.Ticker(t)
                    target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                    chain = tk.option_chain(target_date).puts
                    T = max((datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days, 1) / 365
                    chain['delta'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                    matches = chain[(chain['delta'].abs() <= (100-target_prob)/100) & (chain['bid'] > 0)]
                    if not matches.empty:
                        best = matches.sort_values('bid', ascending=False).iloc[0]
                        y_pa = (best['bid'] / best['strike']) * (365 / (T*365)) * 100
                        if y_pa >= min_yield_pa:
                            results.append({'ticker': t, 'yield': y_pa, 'strike': best['strike'], 'rsi': rsi, 'price': price, 'earn': earn})
                except: continue
    if results:
        opp_df = pd.DataFrame(results).sort_values('yield', ascending=False).head(12)
        cols = st.columns(4)
        for idx, row in enumerate(opp_df.to_dict('records')):
            with cols[idx % 4]:
                st.markdown(f"### {row['ticker']}")
                st.metric("Rendite p.a.", f"{row['yield']:.1f}%")
                st.write(f"Strike: **{row['strike']:.1f}$**")
                st.caption(f"Kurs: {row['price']:.2f}$ | RSI: {row['rsi']}")
                if row['earn']: st.warning(f"📅 ER: {row['earn']}")
    else: st.warning("Keine Treffer unter diesen Sicherheits-Vorgaben.")

st.divider()

# --- 5. DEPOT-STATUS ---
st.subheader("💼 Depot-Status")
depot_data = [
    {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0},
    {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
    {"Ticker": "HOOD", "Einstand": 120.0}, {"Ticker": "NVO", "Einstand": 97.0},
    {"Ticker": "RBRK", "Einstand": 70.0}, {"Ticker": "SE", "Einstand": 170.0},
    {"Ticker": "TTD", "Einstand": 102.0}
]
d_cols = st.columns(4)
for i, item in enumerate(depot_data):
    price, _, earn = get_stock_data(item['Ticker'])
    if price:
        diff = (price / item['Einstand'] - 1) * 100
        icon = "🟢" if diff >= 0 else "🟡" if diff > -20 else "🔴"
        with d_cols[i % 4]:
            st.write(f"{icon} **{item['Ticker']}**: {price:.2f}$ ({diff:.1f}%)")

st.divider()

# --- 6. EINZEL-CHECK MIT AMPEL ---
st.subheader("🔍 Experten Einzel-Check")
c_type, c_tick = st.columns([1, 2])
with c_type: mode = st.radio("Optionstyp", ["put", "call"], horizontal=True)
with c_tick: t_in = st.text_input("Ticker Symbol", value="HOOD").upper()

if t_in:
    price, dates, earn = get_stock_data(t_in)
    if price and dates:
        rsi, trend_ok = get_safety_metrics(t_in)
        m1, m2, m3 = st.columns(3)
        m1.metric("Kurs", f"{price:.2f}$")
        m2.metric("RSI (14d)", rsi)
        m3.metric("Trend", "Stabil" if trend_ok else "Schwach")
        
        d_sel = st.selectbox("Laufzeit wählen", dates, index=1 if len(dates)>1 else 0)
        tk = yf.Ticker(t_in)
        try:
            chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
            T = max((datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days, 1) / 365
            df = chain[chain['bid'] > 0].copy()
            if mode == "put":
                df = df[df['strike'] < price].sort_values('strike', ascending=False)
            else:
                df = df[df['strike'] > price].sort_values('strike', ascending=True)
            
            for _, opt in df.head(6).iterrows():
                delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, option_type=mode)
                abs_delta = abs(delta)
                prob = (1 - abs_delta) * 100
                ampel = "🟢" if abs_delta <= 0.15 else "🟡" if abs_delta <= 0.30 else "🔴"
                
                with st.expander(f"{ampel} Strike {opt['strike']:.1f}$ | Prämie: {opt['bid']:.2f}$ | OTM: {prob:.1f}%"):
                    ca, cb = st.columns(2)
                    ca.write(f"💰 Cash: **{opt['bid']*100:.0f}$**")
                    ca.write(f"🛡️ Puffer: **{(abs(opt['strike']-price)/price)*100:.1f}%**")
                    cb.write(f"📉 Delta: **{abs_delta:.2f}**")
                    cb.write(f"💼 Kapital: **{opt['strike']*100:,.0f}$**")
        except: st.error("Laufzeit-Daten nicht verfügbar.")

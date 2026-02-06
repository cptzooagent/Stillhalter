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
    if option_type == 'call':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1

# --- 2. SICHERHEITS-LOGIK (Kein Caching von Objekten) ---
def get_safety_metrics(symbol):
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period="3mo")
        if len(hist) < 20: return 50.0, True
        
        # RSI 14 Tage
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Trend (SMA 50)
        sma50 = hist['Close'].tail(50).mean()
        trend_ok = hist['Close'].iloc[-1] > (sma50 * 0.95)
        return round(float(rsi), 1), bool(trend_ok)
    except:
        return 50.0, True

@st.cache_data(ttl=900)
def get_stock_data(symbol):
    """Extrahiert Preise, Daten und Earnings ohne Ticker-Objekt-Fehler."""
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
    except:
        return None, [], ""

# --- UI: SIDEBAR ---
st.sidebar.header("🛡️ Strategie & Sicherheit")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)
rsi_min = st.sidebar.slider("Minimum RSI (Kein freier Fall)", 20, 45, 30)

st.sidebar.subheader("💰 Preis-Filter")
min_stock_p = st.sidebar.number_input("Mindestkurs ($)", value=40)
max_stock_p = st.sidebar.number_input("Maximalkurs ($)", value=600)

# --- HAUPTBEREICH ---
st.title("🛡️ CapTrader AI Market Guard Pro")

# SEKTION 1: SCANNER
if st.button("🚀 High-Safety Scan starten"):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "AMD", "NFLX", "COIN", "PLTR", "HOOD", "SQ", "MSTR", "SMCI", "UBER", "ABNB", "DIS", "PYPL"]
    results = []
    prog = st.progress(0)
    status = st.empty()
    
    for i, t in enumerate(watchlist):
        status.text(f"Analysiere {t}...")
        prog.progress((i + 1) / len(watchlist))
        price, dates, earn = get_stock_data(t)
        
        if price and min_stock_p <= price <= max_stock_p and dates:
            rsi, trend_ok = get_safety_metrics(t)
            if rsi >= rsi_min and trend_ok:
                try:
                    tk = yf.Ticker(t)
                    target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                    chain = tk.option_chain(target_date).puts
                    T = max((datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days, 1) / 365
                    max_delta = (100 - target_prob) / 100
                    
                    chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                    safe_opts = chain[(chain['delta_val'].abs() <= max_delta) & (chain['bid'] > 0)].copy()
                    
                    if not safe_opts.empty:
                        best = safe_opts.sort_values('bid', ascending=False).iloc[0]
                        y_pa = (best['bid'] / best['strike']) * (365 / (T*365)) * 100
                        if y_pa >= min_yield_pa:
                            results.append({
                                'ticker': t, 'yield': y_pa, 'strike': best['strike'], 
                                'rsi': rsi, 'price': price, 'earn': earn, 
                                'bid': best['bid'], 'capital': best['strike'] * 100
                            })
                except: continue

    status.text("Scan beendet!")
    if results:
        opp_df = pd.DataFrame(results).sort_values('yield', ascending=False).head(12)
        cols = st.columns(4)
        for idx, row in enumerate(opp_df.to_dict('records')):
            with cols[idx % 4]:
                st.markdown(f"### {row['ticker']}")
                st.metric("Rendite p.a.", f"{row['yield']:.1f}%")
                st.write(f"Strike: **{row['strike']:.1f}$**")
                st.caption(f"Kurs: {row['price']:.2f}$ | RSI: {row['rsi']}")
                if row['earn']: st.warning(f"⚠️ ER: {row['earn']}")
                st.info(f"💼 Bedarf: {row['capital']:,.0f}$")
    else: st.warning("Keine Treffer gefunden. Versuche RSI oder Rendite zu senken.")

st.write("---")

# SEKTION 2: DEPOT
st.subheader("💼 Depot-Status")
depot_data = [
    {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0},
    {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
    {"Ticker": "HOOD", "Einstand": 120.0}, {"Ticker": "NVO", "Einstand": 97.0},
    {"Ticker": "RBRK", "Einstand": 70.0}, {"Ticker": "SE", "Einstand": 170.0},
    {"Ticker": "TTD", "Einstand": 102.0}
]
p_cols = st.columns(4)
for i, item in enumerate(depot_data):
    price, _, earn = get_stock_data(item['Ticker'])
    if price:
        diff = (price / item['Einstand'] - 1) * 100
        icon = "🟢" if diff >= 0 else "🟡" if diff > -20 else "🔴"
        with p_cols[i % 4]:
            st.write(f"{icon} **{item['Ticker']}**: {price:.2f}$ ({diff:.1f}%)")

st.write("---")

# SEKTION 3: EINZEL-CHECK
st.subheader("🔍 Experten Einzel-Check")
c_type, c_tick = st.columns([1, 2])
with c_type: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c_tick: t_in = st.text_input("Ticker eingeben", value="NVDA").upper()

if t_in:
    price, dates, earn = get_stock_data(t_in)
    if price and dates:
        rsi, trend_ok = get_safety_metrics(t_in)
        m1, m2, m3 = st.columns(3)
        m1.metric("Kurs", f"{price:.2f}$")
        m2.metric("RSI (14d)", rsi)
        m3.metric("Trend-Check", "Stabil" if trend_ok else "Schwach")
        
        d_sel = st.selectbox("Laufzeit wählen", dates, index=1 if len(dates)>1 else 0)
        tk = yf.Ticker(t_in)
        chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
        T = max((datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days, 1) / 365
        
        df = chain[chain['bid'] > 0].copy()
        df = df[df['strike'] < price].sort_values('strike', ascending=False) if mode == "put" else df[df['strike'] > price].sort_values('strike', ascending=True)
        
        for _, opt in df.head(5).iterrows():
            delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, option_type=mode)
            prob = (1 - abs(delta)) * 100
            with st.expander(f"Strike {opt['strike']:.1f}$ | Prämie: {opt['bid']:.2f}$ | Wahrsch: {prob:.1f}%"):
                col_a, col_b = st.columns(2)
                col_a.write(f"💰 Cash-Einnahme: **{opt['bid']*100:.0f}$**")
                col_a.write(f"🛡️ Puffer: **{(abs(opt['strike']-price)/price)*100:.1f}%**")
                col_b.write(f"📉 Delta: **{abs(delta):.2f}**")
                col_b.write(f"💼 Kapital: **{opt['strike']*100:,.0f}$**")

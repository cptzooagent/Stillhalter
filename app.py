import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 1. MATHE: DELTA-BERECHNUNG (Stabilisiert) ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    T = max(T, 0.0001) 
    sigma = max(sigma, 0.0001)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1

# --- 2. DATEN-FUNKTIONEN ---
@st.cache_data(ttl=3600)
def get_combined_watchlist():
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ADBE", "NFLX", 
        "AMD", "INTC", "QCOM", "AMAT", "TXN", "MU", "ISRG", "LRCX", "PANW", "SNPS",
        "LLY", "V", "MA", "JPM", "WMT", "XOM", "UNH", "PG", "ORCL", "COST", 
        "ABBV", "BAC", "KO", "PEP", "CRM", "WFC", "DIS", "CAT", "AXP", "IBM",
        "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI", "MSTR",
        "SMCI", "MELI", "BKNG", "DE", "GS", "MS", "BA", "SBUX", "UBER", "ABNB"
    ]

@st.cache_data(ttl=900)
def get_stock_basics(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        earn_info = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_info = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return price, dates, earn_info
    except: return None, [], ""

# --- 3. KI-REPAIR LOGIK ---
def get_repair_call(ticker, current_price):
    try:
        tk = yf.Ticker(ticker)
        dates = tk.options
        if not dates: return None
        target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
        chain = tk.option_chain(target_date).calls
        T = max((datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days, 1) / 365
        chain['delta'] = chain.apply(lambda r: calculate_bsm_delta(current_price, r['strike'], T, r['impliedVolatility'] or 0.4, 'call'), axis=1)
        best_repair = chain[chain['delta'] <= 0.35].sort_values('delta', ascending=False).iloc[0]
        return {"strike": best_repair['strike'], "bid": best_repair['bid'], "date": target_date}
    except: return None

# --- UI: SEITENLEISTE ---
st.sidebar.header("🛡️ Strategie-Einstellungen")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)

st.sidebar.subheader("💰 Preis-Filter")
min_stock_p = st.sidebar.number_input("Mindestkurs ($)", value=40)
max_stock_p = st.sidebar.number_input("Maximalkurs ($)", value=600)

# --- HAUPTBEREICH ---
st.title("🛡️ CapTrader AI Market Scanner & Guard")

# SEKTION 1: SCANNER
if st.button("🚀 Markt-Scan starten"):
    watchlist = get_combined_watchlist()
    results = []
    prog = st.progress(0)
    status = st.empty()
    
    for i, t in enumerate(watchlist):
        status.text(f"Analysiere {t}...")
        prog.progress((i + 1) / len(watchlist))
        price, dates, earn = get_stock_basics(t)
        
        if price and min_stock_p <= price <= max_stock_p and dates:
            try:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = max((datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days, 1) / 365
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                safe_opts = chain[chain['delta_val'].abs() <= max_delta].copy()
                
                if not safe_opts.empty:
                    days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                    safe_opts['y_pa'] = (safe_opts['bid'] / safe_opts['strike']) * (365 / days) * 100
                    matches = safe_opts[safe_opts['y_pa'] >= min_yield_pa]
                    if not matches.empty:
                        best = matches.sort_values('y_pa', ascending=False).iloc[0]
                        results.append({'ticker': t, 'yield': best['y_pa'], 'strike': best['strike'], 'bid': best['bid'], 'puffer': (abs(best['strike'] - price) / price) * 100, 'delta': abs(best['delta_val']), 'earn': earn, 'price': price, 'capital': best['strike'] * 100})
            except: continue

    status.text("Scan abgeschlossen!")
    if results:
        opp_df = pd.DataFrame(results).sort_values('yield', ascending=False).head(12)
        cols = st.columns(4)
        for idx, row in enumerate(opp_df.to_dict('records')):
            with cols[idx % 4]:
                st.markdown(f"### {row['ticker']}")
                st.metric("Rendite p.a.", f"{row['yield']:.1f}%")
                st.write(f"Strike: **{row['strike']:.1f}$**")
                st.write(f"Puffer: **{row['puffer']:.1f}%**")
                st.info(f"💼 Kapital: {row['capital']:,.0f}$")
    else: st.warning("Keine Treffer. Tipp: Mindestrendite p.a. senken.")

st.write("---") 

# SEKTION 2: DEPOT (MIT KI-REPAIR)
st.subheader("💼 Depot-Status & KI-Analyse")
depot_data = [
    {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0},
    {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
    {"Ticker": "GTM", "Einstand": 17.0}, {"Ticker": "HIMS", "Einstand": 37.0},
    {"Ticker": "HOOD", "Einstand": 120.0}, {"Ticker": "JKS", "Einstand": 50.0},
    {"Ticker": "NVO", "Einstand": 97.0}, {"Ticker": "RBRK", "Einstand": 70.0},
    {"Ticker": "SE", "Einstand": 170.0}, {"Ticker": "TTD", "Einstand": 102.0}
]
p_cols = st.columns(4)
for i, item in enumerate(depot_data):
    price, _, earn = get_stock_basics(item['Ticker'])
    if price:
        diff = (price / item['Einstand'] - 1) * 100
        icon = "🟢" if diff >= 0 else "🟡" if diff > -20 else "🔴"
        with p_cols[i % 4]:
            st.markdown(f"**{item['Ticker']}**: {price:.2f}$ ({diff:.1f}%)")
            if diff < -20:
                st.error("🚨 Repair!")
                repair = get_repair_call(item['Ticker'], price)
                if repair:
                    st.caption(f"💡 Call @{repair['strike']}$ ({repair['date']}) | +{repair['bid']*100:.0f}$")
            if earn: st.caption(f"📅 ER: {earn}")

st.write("---") 

# SEKTION 3: EINZEL-CHECK
st.subheader("🔍 Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: t_in = st.text_input("Ticker eingeben", value="NVDA").upper()

if t_in:
    price, dates, earn = get_stock_basics(t_in)
    if price and dates:
        st.write(f"Aktueller Kurs: **{price:.2f}$**")
        d_sel = st.selectbox("Laufzeit", dates, index=1 if len(dates)>1 else 0)
        tk = yf.Ticker(t_in)
        try:
            chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
            T = max((datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days, 1) / 365
            df = chain[chain['bid'] > 0].copy()
            df = df[df['strike'] < price].sort_values('strike', ascending=False) if mode == "put" else df[df['strike'] > price].sort_values('strike', ascending=True)
            
            for _, opt in df.head(6).iterrows():
                delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, option_type=mode)
                risk = "🟢" if abs(delta) < 0.16 else "🟡" if abs(delta) < 0.31 else "🔴"
                with st.expander(f"{risk} Strike {opt['strike']:.1f}$ | Prämie: {opt['bid']:.2f}$"):
                    st.write(f"Wahrsch. OTM: {(1-abs(delta))*100:.1f}% | Kapitalbedarf: {opt['strike']*100:,.0f}$")
                    st.write(f"Puffer: {(abs(opt['strike']-price)/price)*100:.1f}% | Delta: {abs(delta):.2f}")
        except: st.error("Optionen für diesen Ticker/Laufzeit nicht verfügbar.")

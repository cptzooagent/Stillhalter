import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime
import random

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 1. MATHE: DELTA-BERECHNUNG ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1

# --- 2. DATEN-FUNKTIONEN ---
@st.cache_data(ttl=86400)
def get_auto_watchlist():
    high_yield_base = [
        "TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", 
        "UPST", "HOOD", "SOFI", "MSTR", "AI", "SNOW", "SHOP", "PYPL", "ABNB"
    ]
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt"
        response = pd.read_csv(url, header=None, names=['Ticker'])
        nasdaq_list = response['Ticker'].head(100).tolist()
        return list(set(high_yield_base + nasdaq_list))
    except:
        return high_yield_base

@st.cache_data(ttl=900)
def get_stock_basics(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        
        # Earnings Logik
        has_earnings = False
        e_date = None
        if tk.calendar is not None and 'Earnings Date' in tk.calendar:
            next_e = tk.calendar['Earnings Date'][0].replace(tzinfo=None)
            days_to = (next_e - datetime.now()).days
            if 0 <= days_to <= 14:
                has_earnings = True
                e_date = next_e.strftime('%d.%m.')
                
        return price, dates, has_earnings, e_date
    except:
        return None, [], False, None

# --- UI: SEITENLEISTE ---
st.sidebar.header("🛡️ Strategie-Einstellungen")
target_prob = st.sidebar.slider("Gewünschte Sicherheit (OTM %)", 70, 98, 85)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=10)

# --- HAUPTBEREICH ---
st.title("🛡️ CapTrader AI Market Scanner")
st.write(f"Suche nach Puts mit Delta ≤ **{max_delta:.2f}**.")

# SEKTION 1: AUTOMATISCHER SCANNER
if st.button("🚀 Markt-Scan mit Sicherheits-Filter starten"):
    full_watchlist = get_auto_watchlist()
    scan_list = random.sample(full_watchlist, min(len(full_watchlist), 60)) 
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, t in enumerate(scan_list):
        status_text.text(f"Analysiere {t}...")
        progress_bar.progress((i + 1) / len(scan_list))
        
        price, dates, has_e, e_dt = get_stock_basics(t)
        if price and dates:
            try:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.5), axis=1)
                safe_opts = chain[chain['delta_val'].abs() <= max_delta].copy()
                
                if not safe_opts.empty:
                    best = safe_opts.sort_values('delta_val', ascending=False).iloc[0]
                    days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                    y_pa = (best['bid'] / best['strike']) * (365 / days) * 100
                    puffer = (abs(best['strike'] - price) / price) * 100
                    
                    if y_pa >= min_yield_pa:
                        results.append({
                            'ticker': t, 'yield': y_pa, 'strike': best['strike'], 
                            'bid': best['bid'], 'puffer': puffer, 
                            'delta': abs(best['delta_val']), 'days': days,
                            'has_e': has_e, 'e_dt': e_dt
                        })
            except: continue

    status_text.text("Scan abgeschlossen!")
    if results:
        opp_df = pd.DataFrame(results).sort_values('puffer', ascending=False).head(10)
        cols = st.columns(5)
        for idx, (_, row) in enumerate(opp_df.iterrows()):
            with cols[idx % 5]:
                st.markdown(f"### {row['ticker']}")
                if row['has_e']: st.warning(f"⚠️ Earnings: {row['e_dt']}")
                st.metric("Puffer", f"{row['puffer']:.1f}%")
                st.write(f"💰 Prämie: **{row['bid']:.2f}$**") 
                st.write(f"📈 Yield: **{row['yield']:.1f}% p.a.**")
                st.write(f"🎯 Strike: **{row['strike']:.1f}$**")
    else:
        st.warning("Keine Treffer mit diesen Einstellungen.")

st.write("---") 

# SEKTION 2: DEPOT
st.subheader("💼 Depot-Status")
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
    price, _, has_e, e_dt = get_stock_basics(item['Ticker'])
    if price:
        diff = (price / item['Einstand'] - 1) * 100
        icon = "🟢" if diff >= 0 else "🟡" if diff > -20 else "🔵"
        with p_cols[i % 4]:
            st.write(f"{icon} **{item['Ticker']}**: {price:.2f}$ ({diff:.1f}%)")
            if has_e: st.caption(f"⚠️ Earnings am {e_dt}")

st.write("---") 

# SEKTION 3: EINZEL-CHECK
st.subheader("🔍 Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: t_in = st.text_input("Ticker", value="HOOD").upper()

if t_in:
    price, dates, has_e, e_dt = get_stock_basics(t_in)
    if price and dates:
        if has_e: st.error(f"⚠️ ACHTUNG: Earnings am {e_dt}!")
        st.write(f"Aktueller Kurs: **{price:.2f}$**")
        d_sel = st.selectbox("Laufzeit wählen", dates)
        tk = yf.Ticker(t_in)
        chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
        T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
        df = chain[chain['strike'] < price].sort_values('strike', ascending=False) if mode == "put" else chain[chain['strike'] > price].sort_values('strike', ascending=True)
        
        for _, opt in df.head(6).iterrows():
            delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, option_type=mode)
            risk = "🟢" if abs(delta) < 0.16 else "🟡" if abs(delta) < 0.31 else "🔴"
            
            with st.expander(f"{risk} Strike {opt['strike']:.1f}$ | Prämie: {opt['bid']:.2f}$"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"💰 **Optionspreis:** {opt['bid']:.2f}$")
                    st.write(f"💵 **Cash-Einnahme:** {opt['bid']*100:.0f}$")
                    st.write(f"📊 **Wahrscheinlichkeit OTM:** {(1-abs(delta))*100:.1f}%")
                with col_b:
                    st.write(f"🎯 **Kurs-Puffer:** {(abs(opt['strike']-price)/price)*100:.1f}%")
                    st.write(f"📉 **Delta:** {abs(delta):.2f}")
                    st.write(f"🌊 **Implizite Vola:** {opt['impliedVolatility']*100:.1f}%")

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 1. MATHE: DELTA-BERECHNUNG (BLACK-SCHOLES) ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1

# --- 2. DATEN-FUNKTIONEN (CACHED) ---
@st.cache_data(ttl=86400)
def get_auto_watchlist():
    """Holt eine erweiterte Mischung aus Nasdaq-100 und High-IV Werten."""
    # Wir definieren eine starke Basis-Liste mit volatileren Werten
    # So bist du nicht von Wikipedia abhängig
    high_yield_candidates = [
        "TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", 
        "UPST", "HOOD", "SOFI", "MSTR", "AI", "PLUG", "LCID", "GME", "AMC",
        "SNOW", "SHOP", "U", "NET", "DDOG", "RBLX", "PYPL", "Z", "ABNB"
    ]
    
    try:
        # Versuch, die volle Nasdaq-Liste zu laden
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        # Wir fügen einen 'User-Agent' hinzu, damit Wikipedia uns nicht blockiert
        table = pd.read_html(url, storage_options={'User-Agent': 'Mozilla/5.0'})[4]
        wiki_list = table['Ticker'].tolist()
        # Kombiniere beide Listen und entferne Duplikate
        return list(set(high_yield_candidates + wiki_list))
    except:
        # Wenn Wikipedia blockt, nimm unsere starke Vorauswahl
        return high_yield_candidates

@st.cache_data(ttl=900)
def get_stock_basics(symbol):
    """Holt Kurs und Options-Termine von Yahoo."""
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        return price, dates
    except:
        return None, []

# --- UI START ---
st.title("🛡️ CapTrader AI Market Scanner")
st.caption("Daten: Yahoo Finance (15m Delay) | Scan: Nasdaq-100 | Delta Ziel: 0.15")

# --- SEKTION 1: AUTOMATISCHER MARKT-SCANNER ---
st.subheader("🚀 Top 10 Stillhalter-Chancen (Nasdaq-100)")

if st.button("🔥 Gesamten Markt nach 0.15 Delta scannen"):
    full_watchlist = get_auto_watchlist()
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Scan der ersten 50 Ticker für Stabilität
    scan_list = full_watchlist[:] 
    
    for i, t in enumerate(scan_list):
        status_text.text(f"Analysiere {t} ({i+1}/{len(scan_list)})...")
        progress_bar.progress((i + 1) / len(scan_list))
        
        price, dates = get_stock_basics(t)
        if price and dates:
            try:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                
                # Delta für alle Strikes berechnen
                chain['delta'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.5), axis=1)
                chain['diff'] = (chain['delta'].abs() - 0.15).abs()
                best = chain.sort_values('diff').iloc[0]
                
                days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                y_pa = (best['bid'] / best['strike']) * (365 / days) * 100
                
                if y_pa > 10: # Nur Ergebnisse über 10% p.a.
                    results.append({'ticker': t, 'yield': y_pa, 'strike': best['strike'], 'bid': best['bid'], 'days': days, 'price': price})
            except:
                continue

    status_text.text("Scan abgeschlossen!")
    
    if results:
        opp_df = pd.DataFrame(results).sort_values('yield', ascending=False).head(10)
        cols = st.columns(5)
        for idx, (_, row) in enumerate(opp_df.iterrows()):
            with cols[idx % 5]:
                st.markdown(f"### {row['ticker']}")
                st.metric("Yield p.a.", f"{row['yield']:.1f}%")
                st.write(f"Strike: **{row['strike']:.1f}$**")
                st.write(f"Bid: **{row['bid']:.2f}$**")
                st.caption(f"Kurs: {row['price']:.2f}$ | {row['days']} T.")
    else:
        st.warning("Keine attraktiven Optionen (Delta 0.15) gefunden.")

st.write("---")

# --- SEKTION 2: DEPOT-STATUS ---
st.subheader("💼 Mein Depot & Repair-Ampel")
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
    price, _ = get_stock_basics(item['Ticker'])
    if price:
        diff = (price / item['Einstand'] - 1) * 100
        icon = "🟢" if diff >= 0 else "🟡" if diff > -20 else "🔵"
        with p_cols[i % 4]:
            st.write(f"{icon} **{item['Ticker']}**: {price:.2f}$ ({diff:.1f}%)")

st.write("---")

# --- SEKTION 3: EINZEL-FINDER ---
st.subheader("🔍 Manueller Options-Finder")
f1, f2 = st.columns([1, 2])
with f1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with f2: ticker_input = st.text_input("Ticker", value="HOOD").upper()

if ticker_input:
    price, dates = get_stock_basics(ticker_input)
    if price and dates:
        st.write(f"Kurs: **{price:.2f}$**")
        date = st.selectbox("Laufzeit wählen", dates)
        tk = yf.Ticker(ticker_input)
        chain = tk.option_chain(date).puts if mode == "put" else tk.option_chain(date).calls
        T = (datetime.strptime(date, '%Y-%m-%d') - datetime.now()).days / 365
        
        df = chain[chain['strike'] < price].sort_values('strike', ascending=False) if mode == "put" else chain[chain['strike'] > price].sort_values('strike', ascending=True)
        
        for _, opt in df.head(6).iterrows():
            delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, option_type=mode)
            risk = "🟢" if abs(delta) < 0.16 else "🟡" if abs(delta) < 0.31 else "🔴"
            with st.expander(f"{risk} Strike {opt['strike']:.1f}$ | Bid: {opt['bid']:.2f}$"):
                c1, c2 = st.columns(2)
                c1.write(f"**Delta:** {abs(delta):.2f}")
                c1.write(f"**Puffer:** {(abs(opt['strike']-price)/price)*100:.1f}%")
                c2.write(f"**Bid/Ask:** {opt['bid']:.2f}$ / {opt['ask']:.2f}$")



import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime
import random

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 1. MATHE: DELTA & WAHRSCHEINLICHKEIT ---
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
        return price, dates
    except:
        return None, []

# --- UI START ---
st.title("🛡️ CapTrader AI Market Scanner")
st.caption("Fokus: Maximale Verfallswahrscheinlichkeit & Sicherheit")

# --- FILTER-BEREICH ---
st.sidebar.header("⚙️ Strategie-Filter")
target_prob = st.sidebar.slider("Mind. Wahrscheinlichkeit OTM (%)", 70, 95, 85)
# Umrechnung: 85% Wahrscheinlichkeit entspricht ca. 0.15 Delta
max_delta = (100 - target_prob) / 100

min_yield = st.sidebar.number_input("Mindestrendite p.a. (%)", value=10)

st.write(f"Suche nach Puts mit Delta ≤ **{max_delta:.2f}** (entspricht ca. **{target_prob}%** Sicherheit).")

# --- SEKTION 1: AUTOMATISCHER MARKT-SCANNER ---
if st.button("🔥 Markt-Scan mit Sicherheits-Filter starten"):
    full_watchlist = get_auto_watchlist()
    scan_list = random.sample(full_watchlist, min(len(full_watchlist), 60)) 
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, t in enumerate(scan_list):
        status_text.text(f"Scanne {t}...")
        progress_bar.progress((i + 1) / len(scan_list))
        
        price, dates = get_stock_basics(t)
        if price and dates:
            try:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                
                # Delta berechnen
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.5), axis=1)
                
                # Filter: Nur Puts mit Delta kleiner oder gleich unserem Sicherheitsziel
                safe_options = chain[chain['delta_val'].abs() <= max_delta].copy()
                
                if not safe_options.empty:
                    # Wir nehmen die Option, die am nächsten am Delta-Limit liegt für max. Prämie
                    best = safe_options.sort_values('delta_val', ascending=False).iloc[0]
                    
                    days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                    y_pa = (best['bid'] / best['strike']) * (365 / days) * 100
                    puffer = (abs(best['strike'] - price) / price) * 100
                    
                    if y_pa >= min_yield:
                        results.append({
                            'ticker': t, 'yield': y_pa, 'strike': best['strike'], 
                            'bid': best['bid'], 'days': days, 'price': price, 
                            'puffer': puffer, 'delta': abs(best['delta_val'])
                        })
            except:
                continue

    status_text.text("Scan beendet.")
    if results:
        # Sortierung: Höchster Kurs-Puffer zuerst (maximale Sicherheit)
        opp_df = pd.DataFrame(results).sort_values('puffer', ascending=False).head(10)
        cols = st.columns(5)
        for idx, (_, row) in enumerate(opp_df.iterrows()):
            with cols[idx % 5]:
                st.markdown(f"### {row['ticker']}")
                st.metric("Sicherheit", f"{row['puffer']:.1f}% Puffer")
                st.write(f"Yield p.a.: **{row['yield']:.1f}%**")
                st.write(f"Strike: **{row['strike']:.1f}$**")
                st.caption(f"Delta: {row['delta']:.2f} | {row['days']} T.")
    else:
        st.warning("Keine Optionen gefunden, die diese Sicherheits-Kriterien erfüllen.")

st.write("---")
# (Rest des Depot-Codes bleibt identisch)

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
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

# --- 2. DATEN-FUNKTIONEN ---
@st.cache_data(ttl=900)
def get_stock_basics(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        earn_date = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_date = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return price, dates, earn_date
    except: return None, [], ""

# --- UI: SEITENLEISTE ---
st.sidebar.header("🛡️ Strategie-Einstellungen")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 83)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=40)

st.title("🛡️ CapTrader AI Market Scanner")
st.write(f"Suche nach Puts mit Delta ≤ **{max_delta:.2f}** und Rendite ≥ **{min_yield_pa}%**")

# --- 3. SCANNER (OPTIMIERT FÜR HOHE RENDITEN) ---
if st.button("🚀 Markt-Scan starten"):
    # Watchlist mit Fokus auf High-IV Stocks
    watchlist = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI", "MSTR", "AI", "SNOW", "SHOP", "PYPL", "ABNB", "GME", "AMC"]
    results = []
    prog = st.progress(0)
    status = st.empty()
    
    for i, t in enumerate(watchlist):
        status.text(f"Prüfe {t}...")
        prog.progress((i + 1) / len(watchlist))
        price, dates, earn = get_stock_basics(t)
        
        if price and dates:
            try:
                tk = yf.Ticker(t)
                # Sucht nach dem nächsten Monats-Verfall (ca. 30 Tage)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                
                # Delta für alle Strikes berechnen
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.5), axis=1)
                
                # Nur Optionen im Sicherheitsbereich
                safe_opts = chain[chain['delta_val'].abs() <= max_delta].copy()
                
                if not safe_opts.empty:
                    # Berechne Rendite p.a. für alle sicheren Optionen
                    days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                    safe_opts['y_pa'] = (safe_opts['bid'] / safe_opts['strike']) * (365 / days) * 100
                    
                    # Filter nach Mindestrendite
                    matches = safe_opts[safe_opts['y_pa'] >= min_yield_pa]
                    
                    if not matches.empty:
                        best = matches.sort_values('y_pa', ascending=False).iloc[0]
                        results.append({
                            'ticker': t, 'yield': best['y_pa'], 'strike': best['strike'], 
                            'bid': best['bid'], 'puffer': (abs(best['strike'] - price) / price) * 100, 
                            'delta': abs(best['delta_val']), 'earn': earn
                        })
            except: continue

    status.text("Scan abgeschlossen!")
    if results:
        opp_df = pd.DataFrame(results).sort_values('yield', ascending=False)
        cols = st.columns(4)
        for idx, row in enumerate(opp_df.to_dict('records')):
            with cols[idx % 4]:
                st.markdown(f"### {row['ticker']}")
                if row['earn']: st.warning(f"📅 ER: {row['earn']}")
                st.metric("Rendite p.a.", f"{row['yield']:.1f}%")
                st.write(f"💰 Bid: **{row['bid']:.2f}$** | Puffer: **{row['puffer']:.1f}%**")
                st.write(f"🎯 Strike: **{row['strike']:.1f}$** (Δ {row['delta']:.2f})")
    else:
        st.warning(f"Keine Treffer: 40% Rendite bei {target_prob}% Sicherheit ist aktuell am Markt nicht verfügbar. Probiere 25% Rendite oder 75% Sicherheit.")

# ... (Rest des Codes: Depot & Einzel-Check bleiben gleich)

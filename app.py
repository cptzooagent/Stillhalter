import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader Market Scanner", layout="wide")

# --- 1. MATHE: DELTA-BERECHNUNG ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1

# --- 2. AUTOMATISCHE WATCHLIST (NASDAQ-100) ---
@st.cache_data(ttl=86400) # Nur einmal am Tag aktualisieren
def get_auto_watchlist():
    try:
        # Holt die aktuelle Liste der Nasdaq-100 Ticker von Wikipedia
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        table = pd.read_html(url)[4] # Die Tabelle mit den Tickern
        return table['Ticker'].tolist()
    except:
        # Fallback falls Wikipedia-Download fehlschlägt
        return ["TSLA", "NVDA", "AMD", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AVGO", "COST"]

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
st.caption("Auto-Scan: Nasdaq-100 Index | Ziel: Delta 0.15 | 30 Tage Laufzeit")

# SEKTION 1: DER AUTO-SCANNER
st.subheader("🚀 Top 10 Stillhalter-Chancen (Nasdaq-100)")

if st.button("🔥 Gesamten Markt scannen"):
    full_watchlist = get_auto_watchlist()
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Wir begrenzen den Scan auf die Top 50 (nach Marktkapitalisierung/Relevanz), 
    # um die Ladezeit in Streamlit stabil zu halten (ca. 60-90 Sek)
    scan_list = full_watchlist[:50] 
    
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
                
                # Delta berechnen
                chain['delta'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.5), axis=1)
                chain['diff'] = (chain['delta'].abs() - 0.15).abs()
                best = chain.sort_values('diff').iloc[0]
                
                days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                y_pa = (best['bid'] / best['strike']) * (365 / days) * 100
                
                # Nur attraktive Renditen > 12% p.a. aufnehmen
                if y_pa > 12:
                    results.append({'ticker': t, 'yield': y_pa, 'strike': best['strike'], 'bid': best['bid'], 'days': days, 'price': price})
            except:
                continue

    status_text.text("Scan abgeschlossen!")
    
    if results:
        opp_df = pd.DataFrame(results).sort_values('yield', ascending=False).head(10)
        
        # Darstellung in 2 Reihen à 5 Kacheln
        for row_idx in range(2):
            cols = st.columns(5)
            for col_idx in range(5):
                item_idx = row_idx * 5 + col_idx
                if item_idx < len(opp_df):
                    row = opp_df.iloc[item_idx]
                    with cols[col_idx]:
                        with st.container():
                            st.markdown(f"### {row['ticker']}")
                            st.metric("Yield p.a.", f"{row['yield']:.1f}%")
                            st.write(f"Kurs: {row['price']:.2f}$")
                            st.write(f"**Strike: {row['strike']:.1f}$**")
                            st.write(f"Prämie: **{row['bid']:.2f}$**")
                            st.caption(f"Laufzeit: {row['days']} Tage")
    else:
        st.error("Keine passenden Optionen gefunden. Markt evtl. geschlossen?")

st.write("---")

# SEKTION 2: DEPOT (Bleibt wie gehabt)
st.subheader("💼 Depot & Repair-Status")
# ... (Hier der restliche Depot-Code von oben)

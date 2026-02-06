import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

# --- SETUP ---
st.set_page_config(page_title="Pro Stillhalter Dashboard v2", layout="wide")

# --- 1. MATHE-KERN: DELTA BERECHNUNG ---
def bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    """Berechnet das theoretische Delta (Black-Scholes)."""
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1

# --- 2. DATEN-FUNKTIONEN ---
@st.cache_data(ttl=900) # 15 Minuten Cache für Yahoo Daten
def get_quick_data(symbol):
    try:
        tk = yf.Ticker(symbol)
        # Wir nutzen fast_info für extrem schnelle Kursabfragen
        price = tk.fast_info['last_price']
        return tk, price, tk.options
    except:
        return None, None, []

# --- UI START ---
st.title("🛡️ CapTrader Pro Stillhalter Dashboard")
st.caption("Datenquelle: Yahoo Finance (ca. 15 Min. Verzögerung) | Kein 24h-Delay mehr!")

# 1. DER NEUE SCANNER (OHNE API-LIMITS)
st.subheader("💎 Top 10 High-IV Put Gelegenheiten")
if st.button("🚀 Markt-Scan jetzt starten"):
    watchlist = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI"]
    results = []
    
    with st.spinner("Scanne Märkte (Berechne Deltas live)..."):
        for t in watchlist:
            tk, price, dates = get_quick_data(t)
            if dates:
                # Suche Verfall in ca. 30 Tagen
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                
                # Berechne Delta für jeden Strike
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                chain['delta'] = chain.apply(lambda r: bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.5), axis=1)
                
                # Finde Strike am nächsten an Delta 0.15
                chain['diff'] = (chain['delta'].abs() - 0.15).abs()
                best = chain.sort_values('diff').iloc[0]
                
                days = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days
                y_pa = (best['bid'] / best['strike']) * (365 / max(1, days)) * 100
                
                results.append({'ticker': t, 'yield': y_pa, 'strike': best['strike'], 'days': days, 'bid': best['bid']})

    if results:
        opp_df = pd.DataFrame(results).sort_values('yield', ascending=False).head(10)
        cols = st.columns(5)
        for idx, (_, row) in enumerate(opp_df.iterrows()):
            with cols[idx % 5]:
                with st.container(border=True):
                    st.write(f"**{row['ticker']}**")
                    st.metric("Yield p.a.", f"{row['yield']:.1f}%")
                    st.caption(f"Strike: {row['strike']:.1f}$ | Bid: {row['bid']:.2f}$")
                    st.caption(f"Laufzeit: {row['days']} Tage")

st.divider()

# 2. DEPOT-CHECK
st.subheader("💼 Mein Depot Status")
# Hier deine 12 Ticker
portfolio_tickers = ["AFRM", "ELF", "ETSY", "GTLB", "GTM", "HIMS", "HOOD", "JKS", "NVO", "RBRK", "SE", "TTD"]
# Beispielwerte (hier müsstest du deine Einstandskurse pflegen)
einstand = {"AFRM": 76.0, "HOOD": 120.0, "ELF": 109.0} 

p_cols = st.columns(4)
for i, t in enumerate(portfolio_tickers):
    _, curr, _ = get_quick_data(t)
    if curr:
        price_orig = einstand.get(t, curr) # Fallback falls Ticker nicht in Liste
        diff = (curr / price_orig - 1) * 100
        icon = "🟢" if diff >= 0 else "🟡" if diff > -20 else "🔵"
        with p_cols[i % 4]:
            st.write(f"{icon} **{t}**: {curr:.2f}$ ({diff:.1f}%)")

st.divider()

# 3. OPTIONS-FINDER (MANUELL)
st.subheader("🔍 Präzisions-Finder")
c1, c2, c3 = st.columns([1, 1, 2])
with c1: opt_type = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: search_ticker = st.text_input("Ticker", value="HOOD").upper()

if search_ticker:
    tk, price, dates = get_quick_data(search_ticker)
    if tk:
        st.write(f"Aktueller Kurs: **{price:.2f}$**")
        chosen_date = st.selectbox("Laufzeit wählen", dates)
        
        # Daten abrufen
        chain = tk.option_chain(chosen_date).puts if opt_type == "put" else tk.option_chain(chosen_date).calls
        T = (datetime.strptime(chosen_date, '%Y-%m-%d') - datetime.now()).days / 365
        
        # OTM Filter
        if opt_type == "put":
            df = chain[chain['strike'] < price].sort_values('strike', ascending=False)
        else:
            df = chain[chain['strike'] > price].sort_values('strike', ascending=True)
            
        for _, opt in df.head(6).iterrows():
            # Delta live berechnen
            iv = opt['impliedVolatility'] or 0.4
            calc_delta = bsm_delta(price, opt['strike'], T, iv, option_type=opt_type)
            puffer = (abs(opt['strike'] - price) / price) * 100
            
            risk_color = "🟢" if abs(calc_delta) < 0.16 else "🟡" if abs(calc_delta) < 0.31 else "🔴"
            
            with st.expander(f"{risk_color} Strike {opt['strike']:.1f}$ | Bid: {opt['bid']:.2f}$"):
                st.write(f"**Berechnetes Delta:** {abs(calc_delta):.2f}")
                st.write(f"**Sicherheitspuffer:** {puffer:.1f}%")
                st.write(f"**Implizite Vola:** {iv*100:.1f}%")

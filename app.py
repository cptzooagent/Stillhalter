import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="Pro Stillhalter Dashboard", layout="wide")

# API Keys aus Secrets laden
MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- 1. DATEN-FUNKTIONEN MIT STRENGEM CACHING (1 STUNDE) ---
@st.cache_data(ttl=3600)
def get_cached_expirations(symbol):
    try:
        url = f"https://api.marketdata.app/v1/options/expirations/{symbol}/?token={MD_KEY}"
        r = requests.get(url).json()
        return sorted(r.get('expirations', [])) if r.get('s') == 'ok' else []
    except: return []

@st.cache_data(ttl=3600)
def get_cached_chain(symbol, date_str, side):
    try:
        params = {"token": MD_KEY, "side": side, "expiration": date_str}
        url = f"https://api.marketdata.app/v1/options/chain/{symbol}/"
        r = requests.get(url, params=params).json()
        if r.get('s') == 'ok':
            df = pd.DataFrame({
                'strike': r.get('strike', []), 
                'bid': r.get('bid', r.get('mid', [])), 
                'delta': r.get('delta', [])
            })
            df['delta'] = pd.to_numeric(df['delta'], errors='coerce')
            return df.dropna(subset=['delta'])
    except: return None

# --- 2. DER SAFE-SCAN (SPART API CALLS) ---
@st.cache_data(ttl=3600) # Speichert das gesamte Scanner-Ergebnis für 1 Stunde
def run_cached_market_scan(watchlist):
    results = []
    for t in watchlist:
        dates = get_cached_expirations(t)
        if dates:
            # Suche Datum zwischen 20 und 50 Tagen
            target_date = next((d for d in dates if 20 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.today()).days <= 50), dates[0])
            df = get_cached_chain(t, target_date, "put")
            if df is not None and not df.empty:
                try:
                    df['diff'] = (df['delta'].abs() - 0.15).abs()
                    best = df.sort_values('diff').iloc[0].to_dict()
                    days = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.today()).days
                    y_pa = (best['bid'] / best['strike']) * (365 / max(1, days)) * 100
                    best.update({'ticker': t, 'days': days, 'yield': y_pa})
                    results.append(best)
                except: continue
    return results

def get_live_price(symbol):
    try:
        url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}'
        r = requests.get(url).json()
        return float(r['c']) if 'c' in r and r['c'] != 0 else None
    except: return None

# --- UI START ---
st.title("🛡️ CapTrader Pro Stillhalter Dashboard")

# 1. SCANNER SEKTION
st.subheader("💎 Top 10 High-IV Put Gelegenheiten")
st.info("💡 Scan-Ergebnisse werden 1 Std. gespeichert, um API-Limits zu sparen.")

if st.button("🚀 Markt-Scan jetzt starten"):
    watchlist = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI"]
    with st.spinner("Scanne Märkte... (Dies nutzt ca. 24 API-Anfragen)"):
        opps = run_cached_market_scan(watchlist) # Nutzt das 1-Stunden-Caching
    
    if opps:
        opp_df = pd.DataFrame(opps).sort_values('yield', ascending=False).head(10)
        cols = st.columns(5)
        for idx, (_, row) in enumerate(opp_df.iterrows()):
            with cols[idx % 5]:
                with st.container(border=True):
                    st.write(f"**{row['ticker']}**")
                    st.metric("Yield p.a.", f"{row['yield']:.1f}%")
                    st.caption(f"Strike: {row['strike']:.1f}$ | {row['days']} T.")
    else:
        st.error("Keine Daten empfangen. Eventuell API-Limit erreicht?")

st.divider()

# 2. DEPOT & STRATEGIE
st.subheader("💼 Mein Depot & Strategie")
col_depot, col_legende = st.columns([2, 1])

# Volle Liste deiner 12 Ticker
full_portfolio = [
    {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0}, 
    {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
    {"Ticker": "GTM", "Einstand": 17.0}, {"Ticker": "HIMS", "Einstand": 37.0},
    {"Ticker": "HOOD", "Einstand": 120.0}, {"Ticker": "JKS", "Einstand": 50.0},
    {"Ticker": "NVO", "Einstand": 97.0}, {"Ticker": "RBRK", "Einstand": 70.0},
    {"Ticker": "SE", "Einstand": 170.0}, {"Ticker": "TTD", "Einstand": 102.0}
]

if 'portfolio' not in st.session_state or len(st.session_state.portfolio) < 12:
    st.session_state.portfolio = pd.DataFrame(full_portfolio)

with col_depot:
    st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic", use_container_width=True)

with col_legende:
    with st.container(border=True):
        st.markdown("**ℹ️ Ampel-Legende**")
        st.markdown("🟢 **PROFIT:** Kurs > Einstand.")
        st.markdown("🟡 **REPAIR:** Kurs bis -20%. Call **Delta 0.10**.")
        st.markdown("🔵 **DEEP:** Kurs < -20%. Call **Delta 0.05**.")

# Ampel-Anzeige mit aktuellem Kurs
st.write("---")
p_cols = st.columns(4)
for i, (_, row) in enumerate(st.session_state.portfolio.iterrows()):
    curr = get_live_price(row['Ticker'])
    with p_cols[i % 4]:
        if curr:
            diff = (curr / row['Einstand'] - 1) * 100
            icon = "🟢" if diff >= 0 else "🟡" if diff > -20 else "🔵"
            st.write(f"{icon} **{row['Ticker']}**: {curr:.2f}$ ({diff:.1f}%)")

st.divider()

# 3. OPTIONS FINDER (CACHED)
st.subheader("🔍 Options-Finder")
c1, c2 = st.columns([1, 2])
with c1:
    option_mode = st.radio("Strategie", ["put", "call"], horizontal=True)
with c2:
    find_ticker = st.text_input("Ticker-Symbol", value="HOOD").upper()

if find_ticker:
    live_p = get_live_price(find_ticker)
    if live_p:
        st.write(f"Kurs: **{live_p:.2f}$**")
        all_dates = get_cached_expirations(find_ticker)
        if all_dates:
            chosen_date = st.selectbox("Laufzeit", all_dates)
            chain_df = get_cached_chain(find_ticker, chosen_date, option_mode)
            if chain_df is not None:
                if option_mode == "put":
                    chain_df = chain_df[chain_df['strike'] < live_p].sort_values('strike', ascending=False)
                else:
                    chain_df = chain_df[chain_df['strike'] > live_p].sort_values('strike', ascending=True)
                
                for _, opt in chain_df.head(6).iterrows():
                    d_val = abs(opt['delta'])
                    risk = "🟢" if d_val < 0.16 else "🟡" if d_val < 0.31 else "🔴"
                    with st.expander(f"{risk} Strike {opt['strike']:.1f}$ | Bid: {opt['bid']:.2f}$"):
                        st.write(f"Delta: {d_val:.2f} | Wahrscheinlichkeit: {(1-d_val)*100:.0f}%")

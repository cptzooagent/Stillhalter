import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="Pro Stillhalter Dashboard", layout="wide")

MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- DATA FUNCTIONS (1 STUNDE CACHING) ---
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
            return pd.DataFrame({
                'strike': r['strike'], 
                'bid': r.get('bid', r['mid']), 
                'delta': r.get('delta', [0.0]*len(r['strike']))
            })
    except: return None

def get_live_price(symbol):
    try:
        url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}'
        r = requests.get(url).json()
        return float(r['c']) if 'c' in r and r['c'] != 0 else None
    except: return None

# --- UI START ---
st.title("🛡️ CapTrader Pro Stillhalter Dashboard")

# 1. TOP 10 SCANNER
st.subheader("💎 Top 10 High-IV Put Gelegenheiten")
if st.button("🚀 Markt-Scan jetzt starten"):
    watchlist = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI"]
    opps = []
    p_bar = st.progress(0)
    for i, t in enumerate(watchlist):
        p_bar.progress((i + 1) / len(watchlist))
        dates = get_cached_expirations(t)
        if dates:
            target_date = next((d for d in dates if 20 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.today()).days <= 50), dates[0])
            df = get_cached_chain(t, target_date, "put")
            if df is not None and not df.empty:
                df['diff'] = (df['delta'].abs() - 0.15).abs()
                best = df.sort_values('diff').iloc[0].to_dict()
                days = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.today()).days
                y_pa = (best['bid'] / best['strike']) * (365 / max(1, days)) * 100
                best.update({'ticker': t, 'days': days, 'yield': y_pa})
                opps.append(best)
    
    if opps:
        opp_df = pd.DataFrame(opps).sort_values('yield', ascending=False).head(10)
        cols = st.columns(5)
        for idx, (_, row) in enumerate(opp_df.iterrows()):
            with cols[idx % 5]:
                with st.container(border=True):
                    st.write(f"**{row['ticker']}**")
                    st.metric("Yield p.a.", f"{row['yield']:.1f}%")
                    st.caption(f"Strike: {row['strike']:.1f}$ | {row['days']} T.")

st.divider()

# 2. MEIN DEPOT & AMPEL-LEGENDE
st.subheader("💼 Mein Depot & Repair-Ampel")

# Legende hinzufügen
with st.expander("ℹ️ Ampel-Legende anzeigen"):
    st.markdown("""
    - 🟢 **PROFIT:** Aktueller Kurs liegt über dem Einstandskurs. Alles im grünen Bereich.
    - 🟡 **REPAIR:** Kurs liegt bis zu 20% unter Einstand. Empfehlung: Call mit **Delta 0.10** verkaufen.
    - 🔵 **DEEP REPAIR:** Kurs liegt mehr als 20% unter Einstand. Empfehlung: Call mit **Delta 0.05** verkaufen.
    """)

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

# Kompakte Breite für den Editor
col_edit, _ = st.columns([1, 1])
with col_edit:
    st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic", use_container_width=True)

# Ampel-Anzeige in 3 Spalten
p_cols = st.columns(3)
for i, (_, row) in enumerate(st.session_state.portfolio.iterrows()):
    curr = get_live_price(row['Ticker'])
    with p_cols[i % 3]:
        if curr:
            diff = (curr / row['Einstand'] - 1) * 100
            if diff >= 0:
                icon, label = "🟢", "PROFIT"
            elif diff > -20:
                icon, label = "🟡", "REPAIR"
            else:
                icon, label = "🔵", "DEEP"
            st.write(f"{icon} **{row['Ticker']}**: {diff:.1f}% ({label})")

st.divider()

# 3. OPTIONS FINDER (JETZT AUCH MIT 1-STUNDEN CACHE)
st.subheader("🔍 Options-Finder")
c1, c2 = st.columns([1, 2])
with c1:
    option_mode = st.radio("Strategie wählen", ["put", "call"], horizontal=True)
with c2:
    find_ticker = st.text_input("Ticker eingeben", value="HOOD").upper()

if find_ticker:
    live_p = get_live_price(find_ticker)
    if live_p:
        st.write(f"Aktueller Kurs: **{live_p:.2f}$**")
        # Nutzt get_cached_expirations (1 Std. TTL)
        all_dates = get_cached_expirations(find_ticker)
        if all_dates:
            chosen_date = st.selectbox("Laufzeit wählen", all_dates)
            # Nutzt get_cached_chain (1 Std. TTL)
            chain_df = get_cached_chain(find_ticker, chosen_date, option_mode)
            
            if chain_df is not None:
                if option_mode == "put":
                    chain_df = chain_df[chain_df['strike'] < live_p].sort_values('strike', ascending=False)
                else:
                    chain_df = chain_df[chain_df['strike'] > live_p].sort_values('strike', ascending=True)
                
                for _, opt in chain_df.head(6).iterrows():
                    d_val = abs(opt['delta'])
                    risk_c = "🟢" if d_val < 0.16 else "🟡" if d_val < 0.31 else "🔴"
                    with st.expander(f"{risk_c} Strike {opt['strike']:.1f}$ | Bid: {opt['bid']:.2f}$"):
                        st.write(f"Delta: {d_val:.2f} | Wahrscheinlichkeit: {(1-d_val)*100:.0f}%")

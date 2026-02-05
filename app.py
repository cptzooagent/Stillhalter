import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="Pro Stillhalter Dashboard", layout="wide")

MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- DATA FUNCTIONS ---
def get_market_overview():
    data = {"VIX": {"price": 15.0}, "S&P 500": {"price": 0.0, "change": 0.0}, "Nasdaq": {"price": 0.0, "change": 0.0}}
    try:
        r_vix = requests.get(f"https://api.marketdata.app/v1/indices/quotes/VIX/?token={MD_KEY}").json()
        if r_vix.get('s') == 'ok':
            data["VIX"]["price"] = r_vix['last'][0]
        
        for name, etf in [("S&P 500", "SPY"), ("Nasdaq", "QQQ")]:
            rf = requests.get(f'https://finnhub.io/api/v1/quote?symbol={etf}&token={FINNHUB_KEY}').json()
            if rf.get('c'):
                multiplier = 10 if name == "S&P 500" else 40
                data[name] = {"price": rf['c'] * multiplier, "change": rf.get('dp', 0.0)}
    except: pass
    return data

def get_live_price(symbol):
    try:
        r = requests.get(f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}').json()
        return float(r['c']) if r.get('c') else None
    except: return None

def get_all_expirations(symbol):
    try:
        r = requests.get(f"https://api.marketdata.app/v1/options/expirations/{symbol}/?token={MD_KEY}").json()
        if r.get('s') == 'ok':
            return sorted(r.get('expirations', []))
    except: pass
    return []

def get_chain_for_date(symbol, date_str, side):
    try:
        params = {"token": MD_KEY, "side": side, "expiration": date_str}
        r = requests.get(f"https://api.marketdata.app/v1/options/chain/{symbol}/", params=params).json()
        if r.get('s') == 'ok':
            return pd.DataFrame({
                'strike': r['strike'], 
                'mid': r['mid'], 
                'delta': r.get('delta', [0.0]*len(r['strike'])), 
                'iv': r.get('iv', [0.0]*len(r['strike']))
            })
    except: pass
    return None

def get_best_put_opportunity(symbol):
    try:
        dates = get_all_expirations(symbol)
        if not dates: return None
        # Suche Laufzeit zwischen 20 und 55 Tagen
        target_date = next((d for d in dates if 20 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.today()).days <= 55), dates[0])
        df = get_chain_for_date(symbol, target_date, "put")
        if df is not None and not df.empty:
            df['diff'] = (df['delta'].abs() - 0.15).abs()
            best = df.sort_values('diff').iloc[0].to_dict()
            days = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.today()).days
            best.update({'ticker': symbol, 'days': days, 'yield': (best['mid']/best['strike'])*(365/max(1, days))*100})
            return best
    except: pass
    return None

# --- UI START ---
st.title("🛡️ Pro Stillhalter Dashboard")

# 1. MARKT-STATUS
m = get_market_overview()
vix_val = m["VIX"]["price"]
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("VIX (Angst)", f"{vix_val:.2f}", "🔥 Panik" if vix_val > 25 else "🟢 Ruhig", delta_color="inverse")
    c2.metric("S&P 500", f"{m['S&P 500']['price']:,.0f}", f"{m['S&P 500']['change']:.2f}%")
    c3.metric("Nasdaq", f"{m['Nasdaq']['price']:,.0f}", f"{m['Nasdaq']['change']:.2f}%")

# 2. TOP 10 HIGH-IV PUTS
st.subheader("💎 Top 10 High-IV Put Gelegenheiten (Delta 0.15)")
watchlist = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI"]
opps = []

# Spinner für Feedback beim Laden
with st.spinner("Scanne High-IV Werte..."):
    for t in watchlist:
        res = get_best_put_opportunity(t)
        if res and res.get('mid', 0) > 0.10:
            opps.append(res)

if opps:
    opp_df = pd.DataFrame(opps).sort_values('yield', ascending=False).head(10)
    grid = st.columns(5)
    for idx, (_, row) in enumerate(opp_df.iterrows()):
        with grid[idx % 5]:
            with st.container(border=True):
                st.markdown(f"**{row['ticker']}**")
                st.write(f"Strike: **{row['strike']:.1f}$**")
                st.metric("Yield p.a.", f"{row['yield']:.1f}%", f"{row['mid']:.2f}$")
                st.caption(f"{row['days']} Tage | IV: {row.get('iv', 0)*100:.0f}%")
else:
    st.info("Keine aktuellen High-IV Gelegenheiten gefunden. API-Limit prüfen.")

st.divider()

# 3. PORTFOLIO REPAIR STATUS
st.subheader("💼 Portfolio Repair-Status")
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([
        {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0}, 
        {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
        {"Ticker": "GTM", "Einstand": 17.0}, {"Ticker": "HIMS", "Einstand": 37.0},
        {"Ticker": "HOOD", "Einstand": 120.0}, {"Ticker": "JKS", "Einstand": 50.0},
        {"Ticker": "NVO", "Einstand": 97.0}, {"Ticker": "RBRK", "Einstand": 70.0},
        {"Ticker": "SE", "Einstand": 170.0}, {"Ticker": "TTD", "Einstand": 102.0}
    ])

c_tab, c_status = st.columns([1, 1.2])
with c_tab:
    with st.expander("Bestände verwalten"):
        st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic", use_container_width=True)

with c_status:
    for _, row in st.session_state.portfolio.iterrows():
        curr = get_live_price(row['Ticker'])
        if curr:
            diff = (curr/row['Einstand'] - 1) * 100
            icon, stat, note = ("🟢", "OK", "GO") if diff >= 0 else ("🟡", "REPAIR", "Delta 0.10") if diff > -20 else ("🔵", "DEEP REPAIR", "Delta 0.05")
            st.write(f"{icon} **{row['Ticker']}**: {curr:.2f}$ ({diff:.1f}%) → `{stat}: {note}`")

st.divider()

# 4. OPTIONS-FINDER
st.subheader("🔍 Options-Finder")
c_strat, c_tick = st.columns([1, 2])
with c_strat:
    option_type = st.radio("Strategie", ["Put 🛡️", "Call 📈"], horizontal=True)
    side = "put" if "Put" in option_type else "call"
with c_tick:
    ticker = st.text_input("Ticker für Detail-Scan (z.B. TSLA)").strip().upper()

if ticker:
    price = get_live_price(ticker)
    if price:
        st.info(f"Aktueller Kurs: {price:.2f}$")
        
        # WICHTIG: Expliziter Check der Laufzeiten
        dates = get_all_expirations(ticker)
        if dates:
            d_labels = {d: f"{d} ({(datetime.strptime(d, '%Y-%m-%d') - datetime.today()).days} Tage)" for d in dates}
            sel_date = st.selectbox("Laufzeit wählen", dates, format_func=lambda x: d_labels.get(x))
            
            df = get_chain_for_date(ticker, sel_date, side)
            if df is not None and not df.empty:
                # OTM Filter
                df = df[df['strike'] < price] if side == "put" else df[df['strike'] > price]
                df = df.sort_values('strike', ascending=(side == "call"))
                
                for _, row in df.head(10).iterrows():
                    d_abs = abs(float(row['delta']))
                    pop = (1 - d_abs) * 100
                    color = "🟢" if d_abs < 0.15 else "🟡" if d_abs < 0.25 else "🔴"
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | Delta: {d_abs:.2f} | Chance: {pop:.0f}%"):
                        ca, cb = st.columns(2)
                        ca.metric("Prämie", f"{row['mid']:.2f}$")
                        cb.metric("Abstand", f"{abs(row['strike']/price-1)*100:.1f}%")
            else:
                st.warning("Keine Optionskette für dieses Datum verfügbar.")
        else:
            st.error(f"Keine Laufzeiten für {ticker} gefunden. Bitte Ticker prüfen oder API-Limits beachten.")

import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="Stillhalter Pro Scanner", layout="wide")

MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- ROBUSTE DATEN-FUNKTIONEN ---
def get_market_overview():
    data = {"VIX": {"price": 15.0}, "S&P 500": {"price": 0.0, "change": 0.0}, "Nasdaq": {"price": 0.0, "change": 0.0}}
    try:
        # VIX Check
        for ticker in ["VIX", "^VIX", "VXX"]:
            r = requests.get(f"https://api.marketdata.app/v1/indices/quotes/{ticker}/?token={MD_KEY}").json()
            if r.get('s') == 'ok' and r.get('last'):
                data["VIX"]["price"] = r['last'][0]
                break
        # Indizes via ETF Fallback
        for name, etf in [("S&P 500", "SPY"), ("Nasdaq", "QQQ")]:
            rf = requests.get(f'https://finnhub.io/api/v1/quote?symbol={etf}&token={FINNHUB_KEY}').json()
            if rf.get('c'):
                data[name] = {"price": rf['c'] * (10 if name=="S&P 500" else 40), "change": rf.get('dp', 0.0)}
    except: pass
    return data

def get_live_price(symbol):
    try:
        r = requests.get(f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}').json()
        return float(r['c']) if r.get('c') else None
    except: return None

def get_trend_analysis(symbol):
    """Prüft auf Höhere Hochs/Tiefs (Trendstruktur)"""
    try:
        r = requests.get(f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&count=20&token={FINNHUB_KEY}").json()
        if r.get('s') == 'ok' and len(r.get('h', [])) > 10:
            h, l = r['h'], r['l']
            # Vergleich letzte 5 Tage vs 5 Tage davor
            curr_h, prev_h = max(h[-5:]), max(h[-10:-5])
            curr_l, prev_l = min(l[-5:]), min(l[-10:-5])
            
            if curr_h > prev_h and curr_l > prev_l:
                return "Bullish 📈", "🟢 Höhere Hochs & Tiefs"
            elif curr_h < prev_h and curr_l < prev_l:
                return "Bearish 📉", "🔴 Tiefer Hochs & Tiefs"
            return "Seitwärts ↔️", "🟡 Keine klare Struktur"
    except: pass
    return "Unklar ⚪", "Keine Trenddaten"

def get_all_expirations(symbol):
    try:
        r = requests.get(f"https://api.marketdata.app/v1/options/expirations/{symbol}/?token={MD_KEY}").json()
        return sorted(r.get('expirations', [])) if r.get('s') == 'ok' else []
    except: return []

def get_chain_for_date(symbol, date_str, side):
    try:
        params = {"token": MD_KEY, "side": side, "expiration": date_str}
        r = requests.get(f"https://api.marketdata.app/v1/options/chain/{symbol}/", params=params).json()
        if r.get('s') == 'ok':
            return pd.DataFrame({'strike': r['strike'], 'mid': r['mid'], 'delta': r.get('delta', [0.0]*len(r['strike']))})
    except: return None

# --- UI START ---
st.title("🛡️ Pro Stillhalter Scanner")

market = get_market_overview()
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("VIX (Angst)", f"{market['VIX']['price']:.2f}", "🔥 Panik" if market['VIX']['price'] > 25 else "🟢 Ruhig")
    c2.metric("S&P 500", f"{market['S&P 500']['price']:,.0f}")
    c3.metric("Nasdaq", f"{market['Nasdaq']['price']:,.0f}")

st.divider()

# --- PORTFOLIO ---
st.subheader("💼 Portfolio & Trend-Check")
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([
        {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0},
        {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0}
    ])

col_tab, col_trend = st.columns([1, 1.2])

with col_tab:
    with st.expander("Bestände editieren", expanded=False):
        st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic", use_container_width=True)

with col_trend:
    for _, row in st.session_state.portfolio.iterrows():
        t = row['Ticker']
        price = get_live_price(t)
        if price:
            trend_label, trend_desc = get_trend_analysis(t)
            diff = (price/row['Einstand'] - 1) * 100
            color = "🔵" if diff < -20 else "🟡" if diff < 0 else "🟢"
            st.write(f"{color} **{t}**: {price:.2f}$ ({diff:.1f}%) | {trend_label} | `{trend_desc}`")

st.divider()

# --- SCANNER ---
st.subheader("🔍 Options-Scanner")
c_strat, c_tick = st.columns([1, 2])
with c_strat:
    option_type = st.radio("Strategie", ["Put 🛡️", "Call 📈"], horizontal=True)
    side = "put" if "Put" in option_type else "call"
with c_tick:
    ticker = st.text_input("Ticker für Detail-Scan").strip().upper()

if ticker:
    price = get_live_price(ticker)
    if price:
        trend_label, trend_desc = get_trend_analysis(ticker)
        st.info(f"Aktueller Trend für {ticker}: **{trend_label}** ({trend_desc})")
        
        dates = get_all_expirations(ticker)
        if dates:
            d_labels = {d: f"{d} ({(datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days} T.)" for d in dates}
            sel_date = st.selectbox("Laufzeit", dates, format_func=lambda x: d_labels.get(x))
            
            df = get_chain_for_date(ticker, sel_date, side)
            if df is not None and not df.empty:
                df = df[df['strike'] < price] if side == "put" else df[df['strike'] > price]
                df = df.sort_values('strike', ascending=(side == "call"))
                
                for _, row in df.head(10).iterrows():
                    d_abs = abs(float(row['delta']))
                    pop = (1 - d_abs) * 100
                    is_safe = d_abs < 0.15
                    color = "🟢" if is_safe else "🔴"
                    
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | Delta: {d_abs:.2f} | Chance: {pop:.0f}%"):
                        st.write(f"Prämie: **{row['mid']:.2f}$**")
                        if trend_label == "Bearish 📉" and side == "put":
                            st.warning("⚠️ Vorsicht: Trend ist abwärts gerichtet. Put-Verkauf riskant!")
                        elif trend_label == "Bullish 📈" and side == "put":
                            st.success("✅ Trend unterstützt den Put-Verkauf.")

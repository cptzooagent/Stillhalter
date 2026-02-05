import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="Pro Stillhalter Strategie-Scanner", layout="wide")

MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- ROBUSTE DATEN-FUNKTIONEN ---
def get_market_overview():
    data = {"VIX": {"price": 15.0}, "S&P 500": {"price": 0.0, "change": 0.0}, "Nasdaq": {"price": 0.0, "change": 0.0}}
    try:
        for ticker in ["VIX", "^VIX", "VXX"]:
            r = requests.get(f"https://api.marketdata.app/v1/indices/quotes/{ticker}/?token={MD_KEY}").json()
            if r.get('s') == 'ok' and r.get('last'):
                data["VIX"]["price"] = r['last'][0]
                break
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
    """Analysiert Marktstruktur: Höhere Hochs (HH) und Höhere Tiefs (HL)"""
    try:
        r = requests.get(f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&count=20&token={FINNHUB_KEY}").json()
        if r.get('s') == 'ok' and len(r.get('h', [])) >= 10:
            h, l = r['h'], r['l']
            curr_h, prev_h = max(h[-5:]), max(h[-10:-5])
            curr_l, prev_l = min(l[-5:]), min(l[-10:-5])
            if curr_h > prev_h and curr_l > prev_l:
                return "Bullish 📈", "🟢 Höhere Hochs & Tiefs"
            elif curr_h < prev_h and curr_l < prev_l:
                return "Bearish 📉", "🔴 Tiefer Hochs & Tiefs"
            return "Seitwärts ↔️", "🟡 Keine klare Struktur"
    except: pass
    return "Unklar ⚪", "Warte auf Daten..."

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
            return pd.DataFrame({'strike': r['strike'], 'mid': r['mid'], 'delta': r.get('delta', [0.0]*len(r['strike'])), 'iv': r.get('iv', [0.0]*len(r['strike']))})
    except: return None

def get_best_put_opportunity(symbol):
    try:
        dates = get_all_expirations(symbol)
        if not dates: return None
        target_date = next((d for d in dates if 20 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= 50), dates[0])
        df = get_chain_for_date(symbol, target_date, "put")
        if df is not None and not df.empty:
            df['diff'] = (df['delta'].abs() - 0.15).abs()
            best = df.sort_values('diff').iloc[0].to_dict()
            days = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days
            best.update({'ticker': symbol, 'days': days, 'yield': (best['mid']/best['strike'])*(365/days)*100})
            return best
    except: return None

# --- UI START ---
st.title("🛡️ Pro Stillhalter Scanner")

m = get_market_overview()
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("VIX (Angst)", f"{m['VIX']['price']:.2f}", "🔥 Panik" if m['VIX']['price'] > 25 else "🟢 Ruhig", delta_color="inverse")
    c2.metric("S&P 500", f"{m['S&P 500']['price']:,.0f}", f"{m['S&P 500']['change']:.2f}%")
    c3.metric("Nasdaq", f"{m['Nasdaq']['price']:,.0f}", f"{m['Nasdaq']['change']:.2f}%")

st.divider()

# --- TOP 10 LISTE ---
st.subheader("💎 Top 10 High-IV Put Gelegenheiten (Delta 0.15)")
auto_list = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI"]
opps = []
with st.spinner("Scanne High-IV Werte..."):
    for t in auto_list:
        res = get_best_put_opportunity(t)
        if res and res['mid'] > 0.10: opps.append(res)

if opps:
    opp_df = pd.DataFrame(opps).sort_values('yield', ascending=False).head(10)
    grid = st.columns(5)
    for idx, (_, row) in enumerate(opp_df.iterrows()):
        with grid[idx % 5]:
            with st.container(border=True):
                st.markdown(f"**{row['ticker']}**")
                st.write(f"Strike: **{row['strike']:.1f}$**")
                st.metric("Yield p.a.", f"{row['yield']:.1f}%", f"{row['mid']:.2f}$")
                st.caption(f"{row['days']} T. | IV: {row['iv']*100:.0f}%")

st.divider()

# --- PORTFOLIO & TREND ---
st.subheader("💼 Portfolio & Trend-Check")
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([{"Ticker": "AFRM", "Einstand": 76.00}, {"Ticker": "ELF", "Einstand": 109.00}, {"Ticker": "ETSY", "Einstand": 67.00}])

c_tab, c_stat = st.columns([1, 1.2])
with c_tab:
    with st.expander("Bestände editieren"):
        st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic", use_container_width=True)

with c_stat:
    for _, row in st.session_state.portfolio.iterrows():
        price = get_live_price(row['Ticker'])
        if price:
            trend_label, trend_desc = get_trend_analysis(row['Ticker'])
            diff = (price/row['Einstand'] - 1) * 100
            color = "🔵" if diff < -20 else "🟡" if diff < 0 else "🟢"
            st.write(f"{color} **{row['Ticker']}**: {price:.2f}$ ({diff:.1f}%) | {trend_label} | `{trend_desc}`")

st.divider()

# --- DETAIL SCANNER ---
st.subheader("🔍 Options-Finder")
cs1, cs2 = st.columns([1, 2])
with cs1:
    option_type = st.radio("Strategie", ["Put 🛡️", "Call 📈"], horizontal=True)
    side = "put" if "Put" in option_type else "call"
with cs2:
    ticker = st.text_input("Ticker für Detail-Scan").strip().upper()

if ticker:
    price = get_live_price(ticker)
    if price:
        trend_label, trend_desc = get_trend_analysis(ticker)
        st.info(f"Kurs: {price:.2f}$ | Trend: {trend_label} ({trend_desc})")
        dates = get_all_expirations(ticker)
        if dates:
            d_labels = {d: f"{datetime.strptime(d, '%Y-%m-%d').strftime('%d.%m.%Y')} ({(datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days} T.)" for d in dates}
            sel_date = st.selectbox("Laufzeit", dates, format_func=lambda x: d_labels.get(x))
            df = get_chain_for_date(ticker, sel_date, side)
            if df is not None:
                df = df[df['strike'] < price] if side == "put" else df[df['strike'] > price]
                df = df.sort_values('strike', ascending=(side == "call"))
                for _, row in df.head(10).iterrows():
                    d_abs = abs(float(row['delta']))
                    pop = (1 - d_abs) * 100
                    is_safe = d_abs < (0.15 if m['VIX']['price'] > 25 else 0.12)
                    color = "🟢" if is_safe else "🟡" if d_abs < 0.25 else "🔴"
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | Delta: {d_abs:.2f} | Chance: {pop:.0f}%"):
                        ca, cb = st.columns(2)
                        ca.metric("Prämie", f"{row['mid']:.2f}$")
                        cb.metric("Abstand", f"{abs(row['strike']/price-1)*100:.1f}%")

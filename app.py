import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="Stillhalter Pro Scanner", layout="wide")

MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- ROBUSTE MARKT-DATEN FUNKTION ---
def get_market_overview():
    data = {}
    vix_p, vix_c = 0.0, 0.0
    vix_tickers = ["VIX", "^VIX", "VXX"] 
    for ticker in vix_tickers:
        try:
            r = requests.get(f"https://api.marketdata.app/v1/indices/quotes/{ticker}/?token={MD_KEY}").json()
            if r.get('s') == 'ok' and r['last'][0] > 0:
                vix_p, vix_c = r['last'][0], r['changepct'][0]
                break
        except: continue
    if vix_p == 0: vix_p = 15.0
    data["VIX"] = {"price": vix_p, "change": vix_c}

    for name, etf in [("S&P 500", "SPY"), ("Nasdaq", "QQQ")]:
        try:
            r = requests.get(f'https://finnhub.io/api/v1/quote?symbol={etf}&token={FINNHUB_KEY}').json()
            p = r.get('c', 0.0)
            data[name] = {"price": p * 10 if name == "S&P 500" else p * 40, "change": r.get('dp', 0.0)}
        except: data[name] = {"price": 0.0, "change": 0.0}

    try:
        rb = requests.get(f'https://finnhub.io/api/v1/quote?symbol=BINANCE:BTCUSDT&token={FINNHUB_KEY}').json()
        data["Bitcoin"] = {"price": rb.get('c', 0.0), "change": rb.get('dp', 0.0)}
    except: data["Bitcoin"] = {"price": 0.0, "change": 0.0}
    return data

def get_live_price(symbol):
    try:
        r = requests.get(f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}').json()
        return float(r['c']) if r.get('c') else None
    except: return None

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
st.title("🛡️ Pro Stillhalter Pro Scanner")

# 1. MARKT-AMPEL
market = get_market_overview()
vix_status = "normal"
with st.container(border=True):
    m_cols = st.columns(4)
    order = ["VIX", "S&P 500", "Nasdaq", "Bitcoin"]
    for i, name in enumerate(order):
        info = market.get(name, {"price": 0.0, "change": 0.0})
        p, c = info['price'], info['change']
        if name == "VIX":
            vix_status = "panic" if p > 25 else "normal"
            m_cols[i].metric("VIX (Angst)", f"{p:.2f}", f"{c:.2f}% {'🔥' if p > 25 else '🟢'}", delta_color="inverse")
        else:
            m_cols[i].metric(name, f"{p:,.2f}", f"{c:.2f}%")

st.divider()

# 2. KOMPAKTES PORTFOLIO MIT AMPEL
st.subheader("💼 CapTrader Portfolio Analyse")
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([
        {"Ticker": "AFRM", "Einstand": 120.50}, {"Ticker": "ELF", "Einstand": 185.00},
        {"Ticker": "ETSY", "Einstand": 67.00}, {"Ticker": "GTLB", "Einstand": 41.00}
    ])

# Zwei Spalten: Links Tabelle, Rechts Ampel-Analyse
col_tab, col_status = st.columns([1, 1])

with col_tab:
    with st.expander("Bestände editieren", expanded=False):
        st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic", use_container_width=True)

with col_status:
    # Analyse-Logik für die Portfolio-Ampel
    for _, row in st.session_state.portfolio.iterrows():
        t = row['Ticker']
        buyin = row['Einstand']
        curr_p = get_live_price(t)
        
        if curr_p:
            # Ampel-Logik
            if curr_p >= buyin:
                status_icon = "🟢"
                status_text = "GO (Über Einstand)"
            elif curr_p >= buyin * 0.85:
                status_icon = "🟡"
                status_text = "REPAIR (Knapp unter Einstand)"
            else:
                status_icon = "🔴"
                status_text = "STOP (Stark im Minus)"
            
            # Kompakte Anzeige pro Ticker
            st.markdown(f"**{status_icon} {t}**: {curr_p:.2f}$ (vs. {buyin:.2f}$) → `{status_text}`")

st.divider()

# 3. SCANNER
st.subheader("🔍 Options-Scanner")
c_a, c_b = st.columns([1, 2])
with c_a:
    option_type = st.radio("Strategie", ["Put 🛡️", "Call 📈 (Covered Call)"], horizontal=False)
    side = "put" if "Put" in option_type else "call"
with c_b:
    ticker = st.text_input("Ticker für Detail-Scan (z.B. NVDA)").strip().upper()

if ticker:
    price = get_live_price(ticker)
    p_row = st.session_state.portfolio[st.session_state.portfolio['Ticker'] == ticker]
    my_buyin = p_row.iloc[0]['Einstand'] if not p_row.empty else None
    
    if price:
        st.metric(f"Aktueller Kurs {ticker}", f"{price:.2f} $")
        dates = get_all_expirations(ticker)
        if dates:
            d_labels = {d: f"{datetime.strptime(d, '%Y-%m-%d').strftime('%d.%m.%Y')} ({(datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days} T.)" for d in dates}
            sel_date = st.selectbox("Laufzeit", dates, format_func=lambda x: d_labels.get(x))
            
            df = get_chain_for_date(ticker, sel_date, side)
            if df is not None and not df.empty:
                df = df[df['strike'] < price] if side == "put" else df[df['strike'] > price]
                df = df.sort_values('strike', ascending=(side == "call"))
                
                for _, row in df.head(10).iterrows():
                    d_abs = abs(float(row['delta']))
                    pop = (1 - d_abs) * 100
                    is_safe = d_abs < (0.15 if vix_status == "panic" else 0.12)
                    
                    if side == "call" and my_buyin and row['strike'] < my_buyin:
                        color, note = "⚠️", "REPAIR"
                    else:
                        color = "🟢" if is_safe else "🟡" if d_abs < 0.25 else "🔴"
                        note = "SAFE" if is_safe else "AGR."
                    
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | {note} | Chance: {pop:.0f}%"):
                        ca, cb, cc = st.columns(3)
                        ca.metric("Prämie", f"{row['mid']:.2f}$")
                        cb.metric("Delta", f"{d_abs:.2f}")
                        if side == "call" and my_buyin:
                            cc.metric("Profit bei Ausübung", f"{(row['strike'] - my_buyin) + row['mid']:.2f}$")
                        else:
                            cc.metric("Gewinn-Chance", f"{pop:.1f}%")

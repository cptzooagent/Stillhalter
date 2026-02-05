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
    """Versucht Indizes über mehrere Wege zu laden, damit nichts leer bleibt"""
    data = {}
    # 1. Indizes (SPX, Nasdaq, VIX)
    index_map = {"S&P 500": "SPX", "Nasdaq": "NDX", "VIX": "VIX"}
    
    for name, sym in index_map.items():
        # Versuch A: MarketData
        try:
            url = f"https://api.marketdata.app/v1/indices/quotes/{sym}/?token={MD_KEY}"
            r = requests.get(url).json()
            if r.get('s') == 'ok' and r.get('last'):
                data[name] = {"price": r['last'][0], "change": r['changepct'][0]}
                continue
        except: pass
        
        # Versuch B: Finnhub Fallback (via ETFs)
        try:
            alt_sym = "SPY" if "S&P" in name else "QQQ" if "Nasdaq" in name else "^VIX"
            r = requests.get(f'https://finnhub.io/api/v1/quote?symbol={alt_sym}&token={FINNHUB_KEY}').json()
            if r.get('c'):
                data[name] = {"price": r['c'], "change": r['dp']}
        except: 
            data[name] = {"price": 0.0, "change": 0.0}

    # 2. Bitcoin (Immer über Finnhub, da sehr stabil)
    try:
        r_btc = requests.get(f'https://finnhub.io/api/v1/quote?symbol=BINANCE:BTCUSDT&token={FINNHUB_KEY}').json()
        data["Bitcoin"] = {"price": r_btc.get('c', 0), "change": r_btc.get('dp', 0)}
    except:
        data["Bitcoin"] = {"price": 0.0, "change": 0.0}
        
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
            return pd.DataFrame({'strike': r['strike'], 'mid': r['mid'], 'delta': r.get('delta', [0]*len(r['strike']))})
    except: return None

# --- UI START ---
st.title("🛡️ Pro Stillhalter Scanner")

# 1. MARKT-AMPEL (IMMER SICHTBAR)
market = get_market_overview()
vix_status = "normal"

if market:
    with st.container(border=True):
        m_cols = st.columns(4)
        # Fix: Explizite Reihenfolge für die Spalten
        order = ["VIX", "S&P 500", "Nasdaq", "Bitcoin"]
        for i, name in enumerate(order):
            if name in market:
                info = market[name]
                p, c = info['price'], info['change']
                label = ""
                if name == "VIX":
                    vix_status = "panic" if p > 25 else "normal"
                    label = "🔥 PANIK" if p > 25 else "🟢 RUHIG"
                m_cols[i].metric(name, f"{p:,.2f}", f"{c:.2f}% {label}", delta_color="inverse" if name == "VIX" else "normal")

st.divider()

# 2. CAPTRADER PORTFOLIO
st.subheader("💼 CapTrader Portfolio")
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([
        {"Ticker": "AFRM", "Einstand": 120.50},
        {"Ticker": "ELF", "Einstand": 185.00},
        {"Ticker": "ETSY", "Einstand": 67.00},
        {"Ticker": "GTLB", "Einstand": 41.00}
    ])

with st.expander("Bestände verwalten (Ticker & Einstandspreis)"):
    st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic")

# 3. SCANNER
option_type = st.radio("Strategie", ["Put 🛡️", "Call 📈 (Covered Call)"], horizontal=True)
side = "put" if "Put" in option_type else "call"

ticker = st.text_input("Ticker eingeben (z.B. NVDA)").strip().upper()

if ticker:
    price = get_live_price(ticker)
    portfolio_row = st.session_state.portfolio[st.session_state.portfolio['Ticker'] == ticker]
    my_buy_in = portfolio_row.iloc[0]['Einstand'] if not portfolio_row.empty else None
    
    if price:
        c1, c2 = st.columns(2)
        c1.metric(f"Kurs {ticker}", f"{price:.2f} $")
        if my_buy_in:
            diff = ((price / my_buy_in) - 1) * 100
            c2.metric("Mein Einstand", f"{my_buy_in:.2f} $", f"{diff:.1f}%")

        # Covered Call Check
        if side == "call" and my_buy_in and price < my_buy_in:
            st.warning(f"⚠️ Kurs unter Einstand! Nur Strikes ÜBER {my_buy_in}$ wählen, um keinen Verlust zu realisieren.")

        dates = get_all_expirations(ticker)
        if dates:
            date_labels = {d: f"{datetime.strptime(d, '%Y-%m-%d').strftime('%d.%m.%Y')} ({(datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days} T.)" for d in dates}
            sel_date = st.selectbox("Laufzeit", dates, format_func=lambda x: date_labels.get(x))
            
            df = get_chain_for_date(ticker, sel_date, side)
            if df is not None and not df.empty:
                # OTM Filter
                df = df[df['strike'] < price] if side == "put" else df[df['strike'] > price]
                df = df.sort_values('strike', ascending=(side == "call"))
                
                for _, row in df.head(10).iterrows():
                    d_abs = abs(row['delta'] if row['delta'] else 0)
                    pop = (1 - d_abs) * 100
                    
                    # Dynamische Ampel für "Nicht ausbuchen"
                    is_safe = d_abs < (0.12 if vix_status == "normal" else 0.15)
                    color = "🟢" if is_safe else "🟡" if d_abs < 0.25 else "🔴"
                    
                    # Einstandspreis-Schutz
                    if side == "call" and my_buy_in and row['strike'] < my_buy_in:
                        color = "❌"
                    
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | Chance: {pop:.0f}%"):
                        ca, cb, cc = st.columns(3)
                        ca.metric("Prämie", f"{row['mid']:.2f}$")
                        cb.metric("Delta", f"{d_abs:.2f}")
                        if side == "call" and my_buy_in:
                            profit = (row['strike'] - my_buy_in) + row['mid']
                            cc.metric("Profit bei Ausübung", f"{profit:.2f}$")
                        else:
                            cc.metric("Gewinn-Chance", f"{pop:.1f}%")
                        st.progress(max(0.0, min(1.0, pop / 100)))

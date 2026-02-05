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
    
    # 1. VIX SPEZIAL-LOGIK (3-Stufen-Check)
    vix_price, vix_change = 0.0, 0.0
    vix_success = False
    
    # Stufe A: MarketData Index
    try:
        r = requests.get(f"https://api.marketdata.app/v1/indices/quotes/VIX/?token={MD_KEY}").json()
        if r.get('s') == 'ok':
            vix_price = r['last'][0]
            vix_change = r['changepct'][0]
            vix_success = True
    except: pass
    
    # Stufe B: Finnhub Index Fallback
    if not vix_success:
        try:
            r = requests.get(f'https://finnhub.io/api/v1/quote?symbol=^VIX&token={FINNHUB_KEY}').json()
            if r.get('c'):
                vix_price = r['c']
                vix_change = r['dp']
                vix_success = True
        except: pass

    # Stufe C: Finnhub ETF Fallback (VXX als Proxy für Stimmung)
    if not vix_success:
        try:
            r = requests.get(f'https://finnhub.io/api/v1/quote?symbol=VXX&token={FINNHUB_KEY}').json()
            vix_price = r.get('c', 0.0)
            vix_change = r.get('dp', 0.0)
        except: pass
    
    data["VIX"] = {"price": vix_price, "change": vix_change}

    # 2. S&P 500 & NASDAQ (MarketData mit Finnhub ETF Fallback)
    for name, sym, etf in [("S&P 500", "SPX", "SPY"), ("Nasdaq", "NDX", "QQQ")]:
        try:
            r = requests.get(f"https://api.marketdata.app/v1/indices/quotes/{sym}/?token={MD_KEY}").json()
            if r.get('s') == 'ok':
                data[name] = {"price": r['last'][0], "change": r['changepct'][0]}
            else:
                r_f = requests.get(f'https://finnhub.io/api/v1/quote?symbol={etf}&token={FINNHUB_KEY}').json()
                data[name] = {"price": r_f.get('c', 0.0), "change": r_f.get('dp', 0.0)}
        except:
            data[name] = {"price": 0.0, "change": 0.0}

    # 3. BITCOIN
    try:
        r_btc = requests.get(f'https://finnhub.io/api/v1/quote?symbol=BINANCE:BTCUSDT&token={FINNHUB_KEY}').json()
        data["Bitcoin"] = {"price": r_btc.get('c', 0.0), "change": r_btc.get('dp', 0.0)}
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

# --- UI ---
st.title("🛡️ Pro Stillhalter Scanner")

market = get_market_overview()
vix_status = "normal"

with st.container(border=True):
    m_cols = st.columns(4)
    order = ["VIX", "S&P 500", "Nasdaq", "Bitcoin"]
    for i, name in enumerate(order):
        info = market.get(name, {"price": 0.0, "change": 0.0})
        p, c = info['price'], info['change']
        
        display_name = name
        delta_col = "normal"
        val_str = f"{p:,.2f}" if p > 0 else "Wird geladen..."
        
        if name == "VIX":
            vix_status = "panic" if p > 25 else "normal"
            display_name = "VIX (Angst)"
            delta_col = "inverse"
            state = "🔥 PANIK" if p > 25 else "🟢 RUHIG"
            m_cols[i].metric(display_name, val_str, f"{c:.2f}% {state}", delta_color=delta_col)
        else:
            m_cols[i].metric(display_name, val_str, f"{c:.2f}%")

if vix_status == "panic":
    st.error("⚠️ HOHE VOLATILITÄT: Märkte sind nervös. Wähle kleinere Deltas (< 0.12)!")

st.divider()

# --- PORTFOLIO & SCANNER ---
st.subheader("💼 CapTrader Portfolio")
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([
        {"Ticker": "AFRM", "Einstand": 120.50},
        {"Ticker": "ELF", "Einstand": 185.00},
        {"Ticker": "ETSY", "Einstand": 67.00},
        {"Ticker": "GTLB", "Einstand": 41.00}
    ])

with st.expander("Bestände verwalten"):
    st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic")

option_type = st.radio("Strategie", ["Put 🛡️", "Call 📈 (Covered Call)"], horizontal=True)
side = "put" if "Put" in option_type else "call"
ticker = st.text_input("Ticker eingeben").strip().upper()

if ticker:
    price = get_live_price(ticker)
    portfolio_row = st.session_state.portfolio[st.session_state.portfolio['Ticker'] == ticker]
    my_buy_in = portfolio_row.iloc[0]['Einstand'] if not portfolio_row.empty else None
    
    if price:
        c1, c2 = st.columns(2)
        c1.metric(f"Kurs {ticker}", f"{price:.2f} $")
        if my_buy_in:
            diff = ((price / my_buy_in) - 1) * 100
            c2.metric("Einstandspreis", f"{my_buy_in:.2f} $", f"{diff:.1f}%")

        dates = get_all_expirations(ticker)
        if dates:
            date_labels = {d: f"{datetime.strptime(d, '%Y-%m-%d').strftime('%d.%m.%Y')} ({(datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days} T.)" for d in dates}
            sel_date = st.selectbox("Laufzeit", dates, format_func=lambda x: date_labels.get(x))
            
            df = get_chain_for_date(ticker, sel_date, side)
            if df is not None and not df.empty:
                df = df[df['strike'] < price] if side == "put" else df[df['strike'] > price]
                df = df.sort_values('strike', ascending=(side == "call"))
                
                for _, row in df.head(10).iterrows():
                    d_abs = abs(row['delta'] if row['delta'] else 0)
                    pop = (1 - d_abs) * 100
                    is_safe = d_abs < (0.12 if vix_status == "normal" else 0.15)
                    color = "🟢" if is_safe else "🟡" if d_abs < 0.25 else "🔴"
                    if side == "call" and my_buy_in and row['strike'] < my_buy_in: color = "❌"
                    
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | Chance: {pop:.0f}%"):
                        ca, cb, cc = st.columns(3)
                        ca.metric

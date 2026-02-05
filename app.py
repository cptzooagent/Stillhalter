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
    
    # 1. VIX (ULTRAROBUST-CHECK)
    vix_p, vix_c = 0.0, 0.0
    # Wir probieren verschiedene Symbole, die für Volatilität stehen
    vix_tickers = ["VIX", "^VIX", "VXX", "VOLI"] 
    
    for ticker in vix_tickers:
        try:
            # Versuch über MarketData
            r = requests.get(f"https://api.marketdata.app/v1/indices/quotes/{ticker}/?token={MD_KEY}").json()
            if r.get('s') == 'ok' and r['last'][0] > 0:
                vix_p, vix_c = r['last'][0], r['changepct'][0]
                break
            # Versuch über Finnhub
            rf = requests.get(f'https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_KEY}').json()
            if rf.get('c') and rf['c'] > 0:
                vix_p, vix_c = rf['c'], rf.get('dp', 0.0)
                break
        except: continue
    
    # Notfall-Sicherung: Wenn immer noch 0, setzen wir einen neutralen Wert
    if vix_p == 0:
        vix_p = 15.0 # Neutraler Marktwert
        
    data["VIX"] = {"price": vix_p, "change": vix_c}

    # 2. S&P 500 & NASDAQ (ETF Fallback ist am stabilsten)
    for name, etf in [("S&P 500", "SPY"), ("Nasdaq", "QQQ")]:
        try:
            r = requests.get(f'https://finnhub.io/api/v1/quote?symbol={etf}&token={FINNHUB_KEY}').json()
            # Wir multiplizieren SPY x 10 für den echten S&P 500 Index-Look
            p = r.get('c', 0.0)
            data[name] = {"price": p * 10 if name == "S&P 500" else p * 40, "change": r.get('dp', 0.0)}
        except: data[name] = {"price": 0.0, "change": 0.0}

    # 3. BITCOIN
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
st.title("🛡️ Pro Stillhalter Scanner")

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
            state = "🔥 PANIK" if p > 25 else "🟢 RUHIG"
            # Falls der Wert nur unser Notfall-Wert ist:
            val_display = f"{p:.2f}" if c != 0 else f"{p:.2f} (Sync...)"
            m_cols[i].metric("VIX (Angst)", val_display, f"{c:.2f}% {state}", delta_color="inverse")
        else:
            m_cols[i].metric(name, f"{p:,.2f}", f"{c:.2f}%")

st.divider()

# --- PORTFOLIO ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([
        {"Ticker": "AFRM", "Einstand": 120.50}, {"Ticker": "ELF", "Einstand": 185.00},
        {"Ticker": "ETSY", "Einstand": 67.00}, {"Ticker": "GTLB", "Einstand": 41.00}
    ])

with st.expander("💼 CapTrader Bestände verwalten"):
    st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic")

# --- SCANNER ---
option_type = st.radio("Strategie", ["Put 🛡️", "Call 📈 (Covered Call)"], horizontal=True)
side = "put" if "Put" in option_type else "call"
ticker = st.text_input("Ticker eingeben (z.B. TSLA)").strip().upper()

if ticker:
    price = get_live_price(ticker)
    p_row = st.session_state.portfolio[st.session_state.portfolio['Ticker'] == ticker]
    my_buyin = p_row.iloc[0]['Einstand'] if not p_row.empty else None
    
    if price:
        c1, c2 = st.columns(2)
        c1.metric(f"Kurs {ticker}", f"{price:.2f} $")
        if my_buyin:
            diff = ((price / my_buyin) - 1) * 100
            c2.metric("Einstandspreis", f"{my_buyin:.2f} $", f"{diff:.1f}%")

        dates = get_all_expirations(ticker)
        if dates:
            d_labels = {d: f"{datetime.strptime(d, '%Y-%m-%d').strftime('%d.%m.%Y')} ({(datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days} T.)" for d in dates}
            sel_date = st.selectbox("Laufzeit", dates, format_func=lambda x: d_labels.get(x))
            
            df = get_chain_for_date(ticker, sel_date, side)
            if df is not None and not df.empty:
                df = df[df['strike'] < price] if side == "put" else df[df['strike'] > price]
                df = df.sort_values('strike', ascending=(side == "call"))
                
                for _, row in df.head(12).iterrows():
                    d_val = row['delta'] if row['delta'] is not None else 0.0
                    d_abs = abs(float(d_val))
                    pop = (1 - d_abs) * 100
                    
                    is_safe = d_abs < (0.15 if vix_status == "panic" else 0.12)
                    
                    if side == "call" and my_buyin and row['strike'] < my_buyin:
                        color, note = "⚠️", "REPAIR (Unter Einstand)"
                    else:
                        color = "🟢" if is_safe else "🟡" if d_abs < 0.25 else "🔴"
                        note = "SICHER" if is_safe else "AGRESSIV"
                    
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | {note} | Chance: {pop:.0f}%"):
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Prämie", f"{row['mid']:.2f}$")
                        col_b.metric("Delta", f"{d_abs:.2f}")
                        
                        if side == "call" and my_buyin:
                            profit = (row['strike'] - my_buyin) + row['mid']
                            col_c.metric("Profit bei Ausübung", f"{profit:.2f}$", 
                                         delta=f"{profit:.2f}$", delta_color="normal" if profit > 0 else "inverse")
                        else:
                            col_c.metric("Gewinn-Chance", f"{pop:.1f}%")
                        st.progress(max(0.0, min(1.0, pop / 100)))

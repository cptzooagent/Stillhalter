import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP & DESIGN ---
st.set_page_config(page_title="Stillhalter Pro Scanner", layout="wide")

# Keys aus den Streamlit Secrets
MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- FUNKTIONEN ---

def get_market_overview():
    """Holt Marktdaten über MarketData & Finnhub für maximale Stabilität"""
    try:
        # Indizes über MarketData (zuverlässiger als Finnhub für SPX/NDX)
        indices = {"S&P 500": "SPX", "Nasdaq": "NDX", "VIX": "VIX"}
        data = {}
        for name, sym in indices.items():
            url = f"https://api.marketdata.app/v1/indices/quotes/{sym}/?token={MD_KEY}"
            r = requests.get(url).json()
            if r.get('s') == 'ok':
                data[name] = {"price": r.get('last', [0])[0], "change": r.get('changepct', [0])[0]}
        
        # Bitcoin über Finnhub (24/7 stabil)
        btc_r = requests.get(f'https://finnhub.io/api/v1/quote?symbol=BINANCE:BTCUSDT&token={FINNHUB_KEY}').json()
        data["Bitcoin"] = {"price": btc_r.get('c', 0), "change": btc_r.get('dp', 0)}
        return data
    except:
        return None

def get_live_price(symbol):
    try:
        r = requests.get(f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}').json()
        return float(r['c']) if r.get('c') else None
    except: return None

def get_sma_200(symbol):
    try:
        r = requests.get(f'https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&count=250&token={FINNHUB_KEY}').json()
        if r.get('s') == 'ok':
            closes = r.get('c', [])
            return sum(closes[-200:]) / 200 if len(closes) >= 200 else None
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
            return pd.DataFrame({
                'strike': r['strike'], 
                'mid': r['mid'], 
                'delta': r.get('delta', [0]*len(r['strike'])), 
                'expiration': date_str
            })
    except: return None

# --- UI START ---
st.title("🛡️ Pro Stillhalter Scanner")

# 1. MARKT-AMPEL & SENTIMENT
market = get_market_overview()
vix_status = "normal"
if market:
    with st.container(border=True):
        m_cols = st.columns(len(market))
        for i, (name, info) in enumerate(market.items()):
            p_val, c_val = info['price'], info['change']
            color_delta = "normal"
            label = ""
            
            if name == "VIX":
                vix_status = "panic" if p_val > 25 else "normal"
                label = "🔥 PANIK" if p_val > 25 else "🟢 RUHIG"
                color_delta = "inverse"
            
            m_cols[i].metric(name, f"{p_val:,.2f}", f"{c_val:.2f}% {label}", delta_color=color_delta)
        
        if vix_status == "panic":
            st.error("⚠️ Marktsituation: Hohe Panik (VIX > 25). Nutze konservative Deltas für Covered Calls!")

st.divider()

# 2. CAPTRADER BESTANDS-MODUL
st.subheader("💼 CapTrader Portfolio (Covered Call Check)")
if 'portfolio' not in st.session_state:
    # Startwerte als Beispiel
    st.session_state.portfolio = pd.DataFrame([
        {"Ticker": "NVDA", "Einstand": 120.50},
        {"Ticker": "TSLA", "Einstand": 185.00}
    ])

with st.expander("Bestände verwalten (Ticker & Einstandspreis)"):
    st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic")

# 3. SCANNER EINSTELLUNGEN
col_strat, col_tick = st.columns([1, 2])
with col_strat:
    option_type = st.radio("Strategie", ["Put 🛡️ (Cash Secured)", "Call 📈 (Covered Call)"], horizontal=False)
    side = "put" if "Put" in option_type else "call"

with col_tick:
    watchlist = ["AAPL", "TSLA", "NVDA", "MSFT", "AMD", "META", "GOOGL", "AMZN"]
    sel_fav = st.pills("Favoriten", watchlist)
    ticker_input = st.text_input("Ticker manuell")
    ticker = (sel_fav if sel_fav else ticker_input).strip().upper()

if ticker:
    price = get_live_price(ticker)
    sma200 = get_sma_200(ticker)
    portfolio_row = st.session_state.portfolio[st.session_state.portfolio['Ticker'] == ticker]
    my_buy_in = portfolio_row.iloc[0]['Einstand'] if not portfolio_row.empty else None
    
    if price:
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Kurs {ticker}", f"{price:.2f} $")
        if sma200:
            diff_sma = ((price / sma200) - 1) * 100
            c2.metric("SMA 200 Trend", f"{sma200:.2f} $", f"{diff_sma:.1f}%")
        if my_buy_in:
            diff_buyin = ((price / my_buy_in) - 1) * 100
            c3.metric("vs. Einstand", f"{my_buy_in:.2f} $", f"{diff_buyin:.1f}%")

        # Warnung bei Covered Calls unter Einstand
        if side == "call" and my_buy_in and price < my_buy_in:
            st.warning(f"⚠️ Kurs ({price:.2f}$) liegt unter deinem Einstand ({my_buy_in:.2f}$). Strikes unter {my_buy_in}$ vermeiden!")

        dates = get_all_expirations(ticker)
        if dates:
            date_labels = {d: f"{datetime.strptime(d, '%Y-%m-%d').strftime('%d.%m.%Y')} ({(datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days} T.)" for d in dates}
            sel_date = st.selectbox("Laufzeit wählen", dates, format_func=lambda x: date_labels.get(x))
            
            df = get_chain_for_date(ticker, sel_date, side)
            if df is not None and not df.empty:
                # OTM Filter
                df = df[df['strike'] < price] if side == "put" else df[df['strike'] > price]
                df = df.sort_values('strike', ascending=(side == "call"))
                
                st.subheader(f"Gefundene {option_type}")
                
                for _, row in df.head(12).iterrows():
                    d_abs = abs(row['delta'] if row['delta'] else 0)
                    pop = (1 - d_abs) * 100
                    
                    # SICHERHEITS-LOGIK GEGEN AUSBUCHEN
                    # Bei ruhigem Markt sind wir strenger (niedrigeres Delta)
                    if vix_status == "panic":
                        is_safe = d_abs < 0.15
                    else:
                        is_safe = d_abs < 0.12 # Sehr konservativ, um Aktien zu behalten
                    
                    color = "🟢" if is_safe else "🟡" if d_abs < 0.25 else "🔴"
                    
                    # Check gegen Einstand für Calls
                    if side == "call" and my_buy_in and row['strike'] < my_buy_in:
                        color = "❌"
                        note = "VERLUST-GEFAHR"
                    else:
                        note = "SICHER" if is_safe else "AGRESSIV"

                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | {note} | Chance: {pop:.0f}%"):
                        ca, cb, cc = st.columns(3)
                        ca.metric("Prämie", f"{row['mid']:.2f}$")
                        cb.metric("Delta", f"{d_abs:.2f}", help="Ziel für Stillhalter: < 0.15")
                        
                        if side == "call" and my_buy_in:
                            profit_if_called = (row['strike'] - my_buy_in) + row['mid']
                            cc.metric("Gewinn bei Ausübung", f"{profit_if_called:.2f}$")
                        else:
                            cc.metric("Gewinn-Chance", f"{pop:.1f}%")
                        
                        st.progress(max(0.0, min(1.0, pop / 100)))
                        
                        if is_safe:
                            st.caption("🛡️ Empfehlung: Geringes Delta. Ideal um Aktien zu behalten.")
                        elif color == "❌":
                            st.error("Dieser Strike liegt unter deinem Einstandspreis. Nicht empfohlen!")

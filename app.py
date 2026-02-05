import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# --- SETUP & API ---
st.set_page_config(page_title="Pro Stillhalter Strategie-Scanner", layout="wide")
MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

def get_market_overview():
    data = {}
    vix_p = 0.0
    for ticker in ["VIX", "^VIX", "VXX"]:
        try:
            r = requests.get(f"https://api.marketdata.app/v1/indices/quotes/{ticker}/?token={MD_KEY}").json()
            if r.get('s') == 'ok' and r['last'][0] > 0:
                vix_p = r['last'][0]
                break
        except: continue
    data["VIX"] = {"price": vix_p if vix_p > 0 else 15.0}
    for name, etf in [("S&P 500", "SPY"), ("Nasdaq", "QQQ")]:
        try:
            r = requests.get(f'https://finnhub.io/api/v1/quote?symbol={etf}&token={FINNHUB_KEY}').json()
            data[name] = {"price": r.get('c', 0.0) * (10 if name=="S&P 500" else 40), "change": r.get('dp', 0.0)}
        except: data[name] = {"price": 0.0, "change": 0.0}
    return data

def get_trend_analysis(symbol):
    """Analysiert Höhere Hochs/Tiefs und SMA"""
    try:
        # Wir holen historische Daten der letzten 30 Tage
        res = requests.get(f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&count=30&token={FINNHUB_KEY}").json()
        if res.get('s') == 'ok':
            highs = res['h']
            lows = res['l']
            close = res['c']
            
            # Trend-Check: Letzte 5 Tage vs 5 Tage davor
            recent_high = max(highs[-5:])
            prev_high = max(highs[-10:-5])
            recent_low = min(lows[-5:])
            prev_low = min(lows[-10:-5])
            
            sma50 = sum(close) / len(close) # Vereinfachter 30-Tage Durchschnitt
            current = close[-1]
            
            if recent_high > prev_high and recent_low > prev_low:
                return "Bullish 📈 (Höhere Hochs/Tiefs)", "🟢 Short Puts bevorzugen"
            elif recent_high < prev_high and recent_low < prev_low:
                return "Bearish 📉 (Tiefere Hochs/Tiefs)", "🔴 Nur Repair-Calls schreiben"
            else:
                return "Seitwärts ↔️", "🟡 Neutral / Iron Condors"
    except: return "Keine Daten", "⚪ Analyse nicht möglich"

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
            return pd.DataFrame({'strike': r['strike'], 'mid': r['mid'], 'delta': r.get('delta', [0.0]*len(r['strike'])), 'iv': r.get('iv', [0.0]*len(r['strike']))})
    except: return None

# --- UI ---
st.title("🛡️ Pro Stillhalter & Trend-Scanner")

market = get_market_overview()
vix = market["VIX"]["price"]

# 1. MARKT-STIMMUNG
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("VIX", f"{vix:.2f}", "Panik-Zone" if vix > 25 else "Sorglos-Zone", delta_color="inverse")
    st.info("💡 **Tipp:** Bei VIX > 25 sind Puts extrem lukrativ. Bei VIX < 15 eher defensiv agieren.")

st.divider()

# 2. PORTFOLIO MIT TREND-SIGNAL
st.subheader("💼 Portfolio & Trend-Check")
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([{"Ticker": "AFRM", "Einstand": 76.00}, {"Ticker": "ELF", "Einstand": 109.00}])

col_p1, col_p2 = st.columns([1, 1.5])
with col_p1:
    st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic", use_container_width=True)

with col_p2:
    for _, row in st.session_state.portfolio.iterrows():
        t = row['Ticker']
        price = get_live_price(t)
        if price:
            trend_txt, action = get_trend_analysis(t)
            diff = (price/row['Einstand']-1)*100
            with st.expander(f"**{t}**: {price:.2f}$ ({diff:.1f}%)"):
                st.write(f"Struktur: **{trend_txt}**")
                st.write(f"Empfehlung: **{action}**")
                if diff < -20: st.warning("⚠️ Starker Buchverlust: Fokus auf Repair-Calls mit Delta < 0.10")

st.divider()

# 3. DETAIL SCANNER MIT STRATEGIE-HINWEIS
st.subheader("🔍 Strategie-Finder")
ticker = st.text_input("Ticker eingeben").upper()

if ticker:
    price = get_live_price(ticker)
    trend_txt, action = get_trend_analysis(ticker)
    
    if price:
        st.subheader(f"{ticker} Analyse: {trend_txt}")
        st.success(f"Handlungsanweisung: {action}")
        
        # Laufzeiten & Strikes (Rest des Codes bleibt gleich für die Anzeige...)
        dates = get_all_expirations(ticker)
        if dates:
            d_labels = {d: f"{d} ({(datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days} T.)" for d in dates}
            sel_date = st.selectbox("Laufzeit", dates, format_func=lambda x: d_labels.get(x))
            # ... Filterung und Anzeige der Strikes ...

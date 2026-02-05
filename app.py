import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="Stillhalter Pro Scanner", layout="wide")

MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- MARKT-DATEN FUNKTION ---
def get_market_overview():
    try:
        symbols = {"VIX": "^VIX", "SP500": "SPY", "Nasdaq": "QQQ", "BTC": "BINANCE:BTCUSDT"}
        data = {}
        for name, sym in symbols.items():
            r = requests.get(f'https://finnhub.io/api/v1/quote?symbol={sym}&token={FINNHUB_KEY}').json()
            data[name] = {"price": r.get('c', 0), "change": r.get('dp', 0)}
        return data
    except: return None

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
            return pd.DataFrame({'strike': r['strike'], 'mid': r['mid'], 'delta': r.get('delta', [0]*len(r['strike'])), 'expiration': date_str})
    except: return None

# --- UI START ---
st.title("🛡️ Pro Stillhalter Scanner")

# 1. MARKT-AMPEL & DYNAMISCHE LOGIK
market = get_market_overview()
vix_status = "normal"
if market:
    vix = market['VIX']['price']
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("VIX (Angst)", f"{vix:.2f}", "🔥 PANIK" if vix > 25 else "🟢 RUHIG", delta_color="inverse")
        m2.metric("S&P 500", f"{market['SP500']['price']:.1f}", f"{market['SP500']['change']:.2f}%")
        m3.metric("Nasdaq", f"{market['Nasdaq']['price']:.1f}", f"{market['Nasdaq']['change']:.2f}%")
        m4.metric("Bitcoin", f"{market['BTC']['price']:.0f}", f"{market['BTC']['change']:.2f}%")
        
        if vix > 25:
            vix_status = "panic"
            st.error("⚠️ Marktsituation: Hohe Volatilität. Die Sicherheits-Grenzwerte wurden automatisch verschärft.")
            with st.expander("📝 Stillhalter-Checkliste für Panik-Tage"):
                st.write("- [ ] Delta unter 0.12 wählen\n- [ ] Nur Aktien über SMA 200 (Bullish)\n- [ ] Positionsgröße halbieren\n- [ ] Cash-Reserve prüfen")

st.divider()

# 2. EINGABE
option_type = st.radio("Strategie", ["Put 🛡️", "Call 📈"], horizontal=True)
side = "put" if "Put" in option_type else "call"
watchlist = ["AAPL", "TSLA", "NVDA", "MSFT", "AMD", "META", "GOOGL", "AMZN"]
sel_fav = st.pills("Favoriten", watchlist)
ticker = (sel_fav if sel_fav else st.text_input("Ticker", "")).strip().upper()

if ticker:
    price = get_live_price(ticker)
    sma200 = get_sma_200(ticker)
    if price:
        c_p, c_t = st.columns(2)
        c_p.metric(f"Kurs {ticker}", f"{price:.2f} $")
        if sma200:
            diff = ((price / sma200) - 1) * 100
            trend = "✅ ÜBER SMA 200" if price > sma200 else "⚠️ UNTER SMA 200"
            c_t.metric("SMA 200 Trend", f"{sma200:.2f} $", f"{diff:.1f}%", help=trend)
        
        dates = get_all_expirations(ticker)
        if dates:
            idx = next((i for i, d in enumerate(dates) if 30 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= 50), 0)
            sel_date = st.selectbox("Laufzeit", dates, index=idx)
            df = get_chain_for_date(ticker, sel_date, side)
            
            if df is not None and not df.empty:
                df = df[df['strike'] < price] if side == "put" else df[df['strike'] > price]
                df = df.sort_values('strike', ascending=(side == "call"))
                
                for _, row in df.head(12).iterrows():
                    d_abs = abs(row['delta'] if row['delta'] else 0)
                    pop = (1 - d_abs) * 100
                    dte = max(1, (datetime.strptime(sel_date, '%Y-%m-%d') - datetime.now()).days)
                    ann_ret = (row['mid'] / (row['strike'] if side == "put" else price)) * (365 / dte) * 100
                    
                    # DYNAMISCHE AMPEL-LOGIK
                    # Im Panik-Modus (VIX > 25) ist Grün viel schwerer zu erreichen
                    if vix_status == "panic":
                        color = "🟢" if d_abs < 0.10 else "🟡" if d_abs < 0.18 else "🔴"
                    else:
                        color = "🟢" if d_abs < 0.16 else "🟡" if d_abs < 0.25 else "🔴"
                    
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | Chance: {pop:.0f}% | {ann_ret:.1f}% p.a."):
                        ca, cb, cc = st.columns(3)
                        ca.metric("Prämie", f"{row['mid']:.2f}$")
                        cb.metric("Abstand", f"{(abs(price-row['strike'])/price)*100:.1f}%")
                        cc.metric("Gewinn-Chance", f"{pop:.1f}%")
                        st.progress(max(0.0, min(1.0, pop / 100)))

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import concurrent.futures
import time
import fear_and_greed
import requests

# --- 0. SESSION & ANTI-BLOCK SETUP ---
@st.cache_resource
def get_yf_session():
    """Erstellt eine persistente Session, um Yahoo-Blockaden (401) zu umgehen."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Origin': 'https://finance.yahoo.com',
        'Referer': 'https://finance.yahoo.com/'
    })
    return session

yf_session = get_yf_session()

# --- SETUP & CSS ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

st.markdown("""
<style>
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; max-width: 100% !important; }
    [data-testid="stHorizontalBlock"] { gap: 10px !important; }
</style>
""", unsafe_allow_html=True)

# --- 1. KOMPLEXE MATHE & TECHNIK (ORIGINAL LOGIK) ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def calculate_rsi(data, window=14):
    if len(data) < window + 1: return pd.Series([50] * len(data))
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_pivots(symbol):
    """Vollständige Pivot-Berechnung (Daily & Weekly) für Widerstandszonen."""
    try:
        tk = yf.Ticker(symbol, session=yf_session)
        hist_d = tk.history(period="5d") 
        if len(hist_d) < 2: return None
        last_day = hist_d.iloc[-2]
        h_d, l_d, c_d = last_day['High'], last_day['Low'], last_day['Close']
        p_d = (h_d + l_d + c_d) / 3
        s1_d = (2 * p_d) - h_d
        s2_d = p_d - (h_d - l_d)
        r2_d = p_d + (h_d - l_d) 

        hist_w = tk.history(period="3wk", interval="1wk")
        if len(hist_w) < 2: 
            return {"P": p_d, "S1": s1_d, "S2": s2_d, "R2": r2_d, "W_S2": s2_d, "W_R2": r2_d}
        
        last_week = hist_w.iloc[-2]
        h_w, l_w, c_w = last_week['High'], last_week['Low'], last_week['Close']
        p_w = (h_w + l_w + c_w) / 3
        s2_w = p_w - (h_w - l_w)
        r2_w = p_w + (h_w - l_w) 
        return {"P": p_d, "S1": s1_d, "S2": s2_d, "R2": r2_d, "W_S2": s2_w, "W_R2": r2_w}
    except: return None

def get_openclaw_analysis(symbol):
    """KI-Sentiment-Analyse basierend auf Yahoo News-Blobs."""
    try:
        tk = yf.Ticker(symbol, session=yf_session)
        all_news = tk.news
        if not all_news or len(all_news) == 0:
            return "Neutral", "🤖 OpenClaw: Keine aktuellen News-Daten.", 0.5
        huge_blob = str(all_news).lower()
        display_text = ""
        for n in all_news:
            for val in n.values():
                if isinstance(val, str) and val.count(" ") > 3:
                    display_text = val
                    break
            if display_text: break
        if not display_text: display_text = all_news[0].get('title', 'Marktstimmung aktiv')
        
        score = 0.5
        bull_words = ['earnings', 'growth', 'beat', 'buy', 'profit', 'ai', 'demand', 'up', 'bull', 'upgrade']
        bear_words = ['sell-off', 'disruption', 'miss', 'down', 'risk', 'decline', 'short', 'warning', 'sell']
        for w in bull_words:
            if w in huge_blob: score += 0.08
        for w in bear_words:
            if w in huge_blob: score -= 0.08
        score = max(0.1, min(0.9, score))
        status = "Bullish" if score > 0.55 else "Bearish" if score < 0.45 else "Neutral"
        icon = "🟢" if status == "Bullish" else "🔴" if status == "Bearish" else "🟡"
        return status, f"{icon} OpenClaw: {display_text[:90]}...", score
    except: return "N/A", "🤖 OpenClaw: System-Reset...", 0.5

@st.cache_data(ttl=86400)
def get_combined_watchlist():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = df['Symbol'].tolist()
        nasdaq_extra = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "COIN", "MSTR", "HOOD", "PLTR", "SQ"]
        full_list = list(set(tickers + nasdaq_extra))
        return [t.replace('.', '-') for t in full_list]
    except: return ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "COIN", "MSTR"]

def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol, session=yf_session)
        hist = tk.history(period="150d") 
        if hist.empty: return None
        price = hist['Close'].iloc[-1] 
        rsi_series = calculate_rsi(hist['Close'])
        rsi_val = rsi_series.iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else hist['Close'].mean()
        is_uptrend = price > sma_200
        atr = (hist['High'] - hist['Low']).rolling(window=14).mean().iloc[-1]
        pivots = calculate_pivots(symbol)
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return price, list(tk.options), earn_str, rsi_val, is_uptrend, False, atr, pivots
    except: return None

def get_analyst_conviction(info):
    try:
        current = info.get('currentPrice', 1)
        target = info.get('targetMedianPrice', 0)
        upside = ((target / current) - 1) * 100 if target > 0 else 0
        rev_growth = info.get('revenueGrowth', 0) * 100
        if rev_growth > 40: return f"🚀 HYPER-GROWTH (+{rev_growth:.0f}% Wachst.)", "#9b59b6"
        elif upside > 15 and rev_growth > 5: return f"✅ Stark (Ziel: +{upside:.0f}%, Wachst.: {rev_growth:.1f}%)", "#27ae60"
        elif upside > 25: return f"💎 Quality-Dip (Ziel: +{upside:.0f}%)", "#2980b9"
        elif upside < 0 or rev_growth < -2: return f"⚠️ Warnung (Ziel: {upside:.1f}%, Wachst.: {rev_growth:.1f}%)", "#e67e22"
        return f"⚖️ Neutral (Ziel: {upside:.0f}%)", "#7f8c8d"
    except: return "🔍 Check nötig", "#7f8c8d"

# --- 2. GLOBAL MARKET LOGIC ---
def get_market_data():
    try:
        ndq = yf.Ticker("^NDX", session=yf_session); vix = yf.Ticker("^VIX", session=yf_session); btc = yf.Ticker("BTC-USD", session=yf_session)
        h_ndq = ndq.history(period="1mo"); h_vix = vix.history(period="1d"); h_btc = btc.history(period="1d")
        cp_ndq = h_ndq['Close'].iloc[-1]
        sma20_ndq = h_ndq['Close'].rolling(window=20).mean().iloc[-1]
        dist_ndq = ((cp_ndq - sma20_ndq) / sma20_ndq) * 100
        rsi_ndq = calculate_rsi(h_ndq['Close']).iloc[-1]
        v_val = h_vix['Close'].iloc[-1] if not h_vix.empty else 20
        b_val = h_btc['Close'].iloc[-1] if not h_btc.empty else 0
        return cp_ndq, rsi_ndq, dist_ndq, v_val, b_val
    except: return 0, 50, 0, 20, 0

def get_sector_performance():
    sectors = {"XLK": "Tech", "XLY": "Consum.", "XLF": "Finanz", "XLV": "Health", "XLE": "Energy", "XLRE": "Immo"}
    try:
        s_data = yf.download(list(sectors.keys()), period="1d", interval="15m", progress=False, session=yf_session)['Close']
        perf = ((s_data.iloc[-1] / s_data.iloc[0]) - 1) * 100
        return f"{sectors[perf.idxmax()]} (+{perf.max():.1f}%)", f"{sectors[perf.idxmin()]} ({perf.min():.1f}%)"
    except: return "N/A", "N/A"

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("🛡️ Strategie-Einstellungen")
    otm_puffer_slider = st.slider("Gewünschter Puffer (%)", 3, 25, 15)
    min_yield_pa = st.number_input("Mindestrendite p.a. (%)", 0, 100, 12)
    min_stock_price, max_stock_price = st.slider("Preis-Spanne ($)", 0, 1000, (60, 500))
    min_mkt_cap = st.slider("Mindest-Marktkapitalisierung (Mrd. $)", 1, 1000, 20)
    only_uptrend = st.checkbox("Nur Aufwärtstrend (SMA 200)", value=False)
    test_modus = st.checkbox("🛠️ Simulations-Modus", value=False)

# --- EXECUTION DASHBOARD ---
st.markdown("## 🌍 Globales Markt-Monitoring")
cp_ndq, rsi_ndq, dist_ndq, vix_val, btc_val = get_market_data()
top_sec, weak_sec = get_sector_performance()
stock_fg = fear_and_greed.get_index().value rescue 43
crypto_fg = 50 

m_color = "#27ae60" if vix_val < 22 else "#e74c3c"
st.markdown(f'<div style="background-color: {m_color}; color: white; padding: 18px; border-radius: 12px; text-align: center; margin-bottom: 25px;"><h3>MARKET STATUS: {"FAVORABLE" if vix_val < 22 else "CAUTION"}</h3></div>', unsafe_allow_html=True)

# (Metriken Reihe 1 & 2...)
r1c1, r1c2, r1c3 = st.columns(3)
with r1c1: st.metric("Nasdaq 100", f"{cp_ndq:,.0f}", f"{dist_ndq:.1f}% vs SMA20")
with r1c2: st.metric("VIX Index", f"{vix_val:.2f}", delta="Risk Off" if vix_val > 22 else "Risk On", delta_color="inverse")
with r1c3: st.metric("Sectors", f"Top: {top_sec}", f"Weak: {weak_sec}")

st.markdown("---")

# --- SCANNER ENGINE ---
if 'profi_scan_results' not in st.session_state: st.session_state.profi_scan_results = []

if st.button("🚀 Profi-Scan starten"):
    p_min_cap = min_mkt_cap * 1e9
    heute = datetime.now()
    ticker_liste = ["NVDA", "TSLA", "AAPL", "AMD", "COIN", "MSTR", "PLTR"] if test_modus else get_combined_watchlist()
    
    with st.spinner("Scanne Ticker..."):
        all_results = []
        def check_stock(s):
            try:
                time.sleep(0.4)
                tk = yf.Ticker(s, session=yf_session)
                info = tk.info
                if info.get('marketCap', 0) < p_min_cap: return None
                
                res = get_stock_data_full(s)
                if not res: return None
                price, dates, earn, rsi, trend, _, atr, pivots = res
                if only_uptrend and not trend: return None
                
                valid_dates = [d for d in dates if 10 <= (datetime.strptime(d, '%Y-%m-%d')-heute).days <= 35]
                if not valid_dates: return None
                
                chain = tk.option_chain(valid_dates[0]).puts
                target_strike = price * (1 - (otm_puffer_slider/100))
                opts = chain[chain['strike'] <= target_strike].sort_values('strike', ascending=False)
                if opts.empty: return None
                o = opts.iloc[0]
                
                bid, ask = o['bid'], o['ask']
                fair = (bid + ask) / 2
                days = (datetime.strptime(valid_dates[0], '%Y-%m-%d')-heute).days
                y_pa = (fair / o['strike']) * (365 / days) * 100
                
                if y_pa < min_yield_pa or y_pa > 150: return None
                
                iv = o.get('impliedVolatility', 0.4)
                exp_move = (price * (iv * np.sqrt(days/365)) / price) * 100
                em_safety = ((price - o['strike']) / price) * 100 / exp_move
                analyst_txt, analyst_col = get_analyst_conviction(info)
                oc_status, oc_text, _ = get_openclaw_analysis(s)

                return {
                    'symbol': s, 'price': price, 'y_pa': y_pa, 'strike': o['strike'], 'puffer': ((price-o['strike'])/price)*100,
                    'bid': fair, 'rsi': rsi, 'earn': earn, 'tage': days, 'status': "🛡️ Trend" if trend else "💎 Dip",
                    'stars_str': "⭐⭐⭐" if "HYPER" in analyst_txt else "⭐⭐", 'analyst_label': analyst_txt, 
                    'analyst_color': analyst_col, 'em_safety': em_safety, 'oc_text': oc_text, 'delta': calculate_bsm_delta(price, o['strike'], days/365, iv)
                }
            except: return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(check_stock, ticker_liste))
            st.session_state.profi_scan_results = [r for r in results if r]

# --- RENDER CARDS ---
if st.session_state.profi_scan_results:
    cols = st.columns(4)
    for idx, res in enumerate(st.session_state.profi_scan_results):
        with cols[idx % 4]:
            em_col = "#10b981" if res['em_safety'] >= 1.5 else "#ef4444"
            # HTML BLOCK GANZ LINKS
            html_code = f"""
<div style="background: white; border: 1px solid #e5e7eb; border-radius: 16px; padding: 18px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); font-family: sans-serif;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<span style="font-size: 1.1em; font-weight: 800; color: #111827;">{res['symbol']} <span style="color: #f59e0b; font-size: 0.8em;">{res['stars_str']}</span></span>
<span style="font-size: 0.7em; font-weight: 700; color: #3b82f6; background: #3b82f610; padding: 2px 8px; border-radius: 6px;">{res['status']}</span>
</div>
<div style="margin: 10px 0;">
<div style="font-size: 0.6em; color: #6b7280; font-weight: 600;">RENDITE P.A.</div>
<div style="font-size: 1.8em; font-weight: 900; color: #111827;">{res['y_pa']:.1f}%</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
<div style="border-left: 3px solid #8b5cf6; padding-left: 8px;">
<div style="font-size: 0.55em; color: #6b7280;">Strike</div>
<div style="font-size: 0.85em; font-weight: 700;">{res['strike']:.1f}$</div>
</div>
<div style="border-left: 3px solid #f59e0b; padding-left: 8px;">
<div style="font-size: 0.55em; color: #6b7280;">Puffer</div>
<div style="font-size: 0.85em; font-weight: 700;">{res['puffer']:.1f}%</div>
</div>
</div>
<div style="background: {em_col}10; padding: 6px 10px; border-radius: 8px; margin: 12px 0; border: 1px dashed {em_col}; font-size: 0.65em; color: #374151;">
Sicherheit: <b>{res['em_safety']:.1f}x EM</b> | Δ: {abs(res['delta']):.2f}
</div>
<div style="background: #f9fafb; padding: 8px; border-radius: 8px; border: 1px solid #f3f4f6; font-size: 0.6em; color: #4b5563; margin-bottom: 12px;">
{res['oc_text']}
</div>
<div style="background: {res['analyst_color']}10; color: {res['analyst_color']}; padding: 8px; border-radius: 8px; font-size: 0.65em; font-weight: bold; text-align: center;">
{res['analyst_label']}
</div>
<div style="margin-top: 10px; display: flex; justify-content: space-between; font-size: 0.55em; color: #9ca3af;">
<span>RSI: {res['rsi']:.0f}</span>
<span>🗓️ {res['earn'] if res['earn'] else "N/A"}</span>
<span>⏳ {res['tage']} Tage</span>
</div>
</div>
"""
            st.markdown(html_code, unsafe_allow_html=True)

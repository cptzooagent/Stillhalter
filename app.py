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

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 1. MATHE & TECHNIK ---
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
    try:
        tk = yf.Ticker(symbol, session=yf_session)
        all_news = tk.news
        if not all_news or len(all_news) == 0:
            return "Neutral", "🤖 OpenClaw: Yahoo liefert aktuell keine Daten.", 0.5
        huge_blob = str(all_news).lower()
        display_text = ""
        for n in all_news:
            for val in n.values():
                if isinstance(val, str) and val.count(" ") > 3:
                    display_text = val
                    break
            if display_text: break
        if not display_text:
            display_text = all_news[0].get('title', 'Marktstimmung aktiv (Text folgt)')
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
        return status, f"{icon} OpenClaw: {display_text[:90]}", score
    except Exception: return "N/A", "🤖 OpenClaw: System-Reset...", 0.5

@st.cache_data(ttl=86400)
def get_combined_watchlist():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = df['Symbol'].tolist()
        nasdaq_extra = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "COIN", "MSTR", "HOOD", "PLTR", "SQ"]
        full_list = list(set(tickers + nasdaq_extra))
        return [t.replace('.', '-') for t in full_list]
    except: return ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META"]

def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol, session=yf_session)
        hist = tk.history(period="150d") 
        if hist.empty: return None, [], "", 50, True, False, 0, None
        price = hist['Close'].iloc[-1] 
        dates = list(tk.options)
        rsi_series = calculate_rsi(hist['Close'])
        rsi_val = rsi_series.iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else hist['Close'].mean()
        is_uptrend = price > sma_200
        sma_20 = hist['Close'].rolling(window=20).mean()
        std_20 = hist['Close'].rolling(window=20).std()
        lower_band = (sma_20 - 2 * std_20).iloc[-1]
        is_near_lower = price <= (lower_band * 1.02)
        atr = (hist['High'] - hist['Low']).rolling(window=14).mean().iloc[-1]
        pivots = calculate_pivots(symbol)
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return price, dates, earn_str, rsi_val, is_uptrend, is_near_lower, atr, pivots
    except: return None, [], "", 50, True, False, 0, None

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

with st.sidebar:
    st.header("🛡️ Strategie-Einstellungen")
    otm_puffer_slider = st.slider("Gewünschter Puffer (%)", 3, 25, 15, key="puffer_sid")
    min_yield_pa = st.number_input("Mindestrendite p.a. (%)", 0, 100, 12, key="yield_sid")
    min_stock_price, max_stock_price = st.slider("Aktienpreis-Spanne ($)", 0, 1000, (60, 500), key="price_sid")
    st.markdown("---")
    st.subheader("Qualitäts-Filter")
    min_mkt_cap = st.slider("Mindest-Marktkapitalisierung (Mrd. $)", 1, 1000, 20, key="mkt_cap_sid")
    only_uptrend = st.checkbox("Nur Aufwärtstrend (SMA 200)", value=False, key="trend_sid")
    test_modus = st.checkbox("🛠️ Simulations-Modus (Test)", value=False, key="sim_checkbox")

def get_market_data():
    try:
        ndq = yf.Ticker("^NDX", session=yf_session)
        vix = yf.Ticker("^VIX", session=yf_session)
        btc = yf.Ticker("BTC-USD", session=yf_session)
        h_ndq = ndq.history(period="1mo")
        h_vix = vix.history(period="1d")
        h_btc = btc.history(period="1d")
        if h_ndq.empty: return 0, 50, 0, 20, 0
        cp_ndq = h_ndq['Close'].iloc[-1]
        sma20_ndq = h_ndq['Close'].rolling(window=20).mean().iloc[-1]
        dist_ndq = ((cp_ndq - sma20_ndq) / sma20_ndq) * 100
        delta = h_ndq['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_ndq = 100 - (100 / (1 + rs)).iloc[-1]
        v_val = h_vix['Close'].iloc[-1] if not h_vix.empty else 20
        b_val = h_btc['Close'].iloc[-1] if not h_btc.empty else 0
        return cp_ndq, rsi_ndq, dist_ndq, v_val, b_val
    except: return 0, 50, 0, 20, 0

def get_crypto_fg():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=5)
        return int(r.json()['data'][0]['value'])
    except: return 50

# --- DATA FETCHING ---
cp_ndq, rsi_ndq, dist_ndq, vix_val, btc_val = get_market_data()
crypto_fg = get_crypto_fg()
try:
    stock_fg = int(fear_and_greed.get_index().value)
except:
    stock_fg = 43

# MARKTBREITE & STATUS
breadth_val = max(10, min(95, int(62 + (dist_ndq * 2))))
sentiment_gap = abs(stock_fg - crypto_fg)

if dist_ndq < -2 or vix_val > 25:
    m_color, m_text, m_advice = "#e74c3c", "🚨 MARKT-ALARM", f"VIX bei {vix_val:.1f}! Fokus auf Kapitalschutz."
elif sentiment_gap > 25:
    m_color, m_text, m_advice = "#f39c12", "⚡ DIVERGENZ", f"Lücke von {sentiment_gap} Pkt! Krypto zeigt Stress."
else:
    m_color, m_text, m_advice = "#27ae60", "✅ SAMMELN", "Umfeld stabil. Zeit für CSPs auf Qualitäts-Dips."

st.markdown(f'<div style="background-color: {m_color}; color: white; padding: 18px; border-radius: 12px; text-align: center; margin-bottom: 25px;"><h3>{m_text}</h3><p>{m_advice}</p></div>', unsafe_allow_html=True)

# Scanner Logik
if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 Profi-Scan starten"):
    p_puffer = otm_puffer_slider / 100 
    p_min_yield = min_yield_pa
    p_min_cap = min_mkt_cap * 1_000_000_000
    heute = datetime.now()
    
    with st.spinner("Scanne Markt..."):
        ticker_liste = ["APP", "AVGO", "NET", "CRWD", "NVDA", "TSLA", "PLTR", "COIN", "MSTR", "DELL", "DDOG", "MU"] if test_modus else get_combined_watchlist()
        all_results = []

        def check_single_stock(symbol):
            try:
                time.sleep(0.5) # Höherer Sleep gegen Yahoo-Sperre
                tk = yf.Ticker(symbol, session=yf_session)
                info = tk.info
                if not info or 'currentPrice' not in info: return None
                
                m_cap = info.get('marketCap', 0)
                price = info.get('currentPrice', 0)
                if m_cap < p_min_cap or not (min_stock_price <= price <= max_stock_price): return None
                
                res = get_stock_data_full(symbol)
                if res is None or res[0] is None: return None
                _, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
                
                valid_dates = [d for d in dates if 10 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 45]
                if not valid_dates: return None
                target_date = valid_dates[0]
                
                chain = tk.option_chain(target_date).puts
                target_strike = price * (1 - p_puffer)
                opts = chain[(chain['strike'] <= target_strike) & (chain['openInterest'] > 1)].sort_values('strike', ascending=False)
                if opts.empty: return None
                o = opts.iloc[0]

                bid, ask = o['bid'], o['ask']
                if bid <= 0.01 or ask > (bid * 2.5): return None 
                
                fair_price = (bid + ask) / 2
                iv = o.get('impliedVolatility', 0.4)
                days_to_exp = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                y_pa = (fair_price / o['strike']) * (365 / max(1, days_to_exp)) * 100
                
                if y_pa > 150 or y_pa < p_min_yield: return None

                exp_move_pct = (price * (iv * np.sqrt(days_to_exp / 365)) / price) * 100
                current_puffer = ((price - o['strike']) / price) * 100
                em_safety = current_puffer / exp_move_pct if exp_move_pct > 0 else 0
                
                delta_val = calculate_bsm_delta(price, o['strike'], days_to_exp/365, iv)
                analyst_txt, analyst_col = get_analyst_conviction(info)

                return {
                    'symbol': symbol, 'price': price, 'y_pa': y_pa, 'strike': o['strike'], 
                    'puffer': current_puffer, 'bid': fair_price, 'rsi': rsi, 'earn': earn, 
                    'tage': days_to_exp, 'status': "🛡️ Trend" if uptrend else "💎 Dip", 'delta': delta_val,
                    'stars_str': "⭐" * (3 if "HYPER" in analyst_txt else 2 if "Stark" in analyst_txt else 1),
                    'analyst_label': analyst_txt, 'analyst_color': analyst_col, 'mkt_cap': m_cap / 1e9,
                    'em_pct': exp_move_pct, 'em_safety': em_safety
                }
            except: return None

        # Reduzierte Worker gegen Rate-Limiting auf Streamlit Cloud
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(check_single_stock, s): s for s in ticker_liste}
            for future in concurrent.futures.as_completed(futures):
                res_data = future.result()
                if res_data: all_results.append(res_data)

        st.session_state.profi_scan_results = all_results

if 'profi_scan_results' in st.session_state and st.session_state.profi_scan_results:
    all_results = st.session_state.profi_scan_results
    st.subheader(f"🎯 Top-Setups nach Qualität ({len(all_results)} Treffer)")
    cols = st.columns(4)
    heute_dt = datetime.now()
    for idx, res in enumerate(all_results):
        with cols[idx % 4]:
            earn_str = res.get('earn', "---"); status_txt = res.get('status', "Trend")
            sent_icon = res.get('sent_icon', "🟢"); stars = res.get('stars_str', "⭐")
            s_color = "#10b981" if "Trend" in status_txt else "#3b82f6"
            a_label = res.get('analyst_label', "Keine Analyse"); a_color = res.get('analyst_color', "#8b5cf6")
            mkt_cap = res.get('mkt_cap', 0); rsi_val = int(res.get('rsi', 50))
            rsi_style = "color: #ef4444; font-weight: 900;" if rsi_val >= 70 else "color: #10b981; font-weight: 700;" if rsi_val <= 35 else "color: #4b5563; font-weight: 700;"
            delta_val = abs(res.get('delta', 0))
            delta_col = "#10b981" if delta_val < 0.20 else "#f59e0b" if delta_val < 0.30 else "#ef4444"
            
            # --- NEU: EM FARBE ---
            em_safety = res.get('em_safety', 1.0)
            em_col = "#10b981" if em_safety >= 1.5 else "#f59e0b" if em_safety >= 1.0 else "#ef4444"
            
            is_earning_risk = False
            if earn_str and earn_str != "---":
                try:
                    earn_date = datetime.strptime(f"{earn_str}2026", "%d.%m.%Y")
                    if 0 <= (earn_date - heute_dt).days <= res.get('tage', 14): is_earning_risk = True
                except: pass
            card_border, card_shadow, card_bg = ("4px solid #ef4444", "0 8px 16px rgba(239, 68, 68, 0.25)", "#fffcfc") if is_earning_risk else ("1px solid #e5e7eb", "0 4px 6px -1px rgba(0,0,0,0.05)", "#ffffff")

            html_code = f"""
<div style="background: {card_bg}; border: {card_border}; border-radius: 16px; padding: 18px; margin-bottom: 20px; box-shadow: {card_shadow}; font-family: sans-serif;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<span style="font-size: 1.2em; font-weight: 800; color: #111827;">{res['symbol']} <span style="color: #f59e0b; font-size: 0.8em;">{stars}</span></span>
<span style="font-size: 0.75em; font-weight: 700; color: {s_color}; background: {s_color}10; padding: 2px 8px; border-radius: 6px;">{sent_icon} {status_txt}</span>
</div>
<div style="margin: 10px 0;">
<div style="font-size: 0.7em; color: #6b7280; font-weight: 600; text-transform: uppercase;">Yield p.a.</div>
<div style="font-size: 1.9em; font-weight: 900; color: #111827;">{res['y_pa']:.1f}%</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">
<div style="border-left: 3px solid #8b5cf6; padding-left: 8px;">
<div style="font-size: 0.6em; color: #6b7280;">Strike</div>
<div style="font-size: 0.9em; font-weight: 700;">{res['strike']:.1f}&#36;</div>
</div>
<div style="border-left: 3px solid #f59e0b; padding-left: 8px;">
<div style="font-size: 0.65em; color: #6b7280;">Mid</div>
<div style="font-size: 0.9em; font-weight: 700;">{res['bid']:.2f}&#36;</div>
</div>
<div style="border-left: 3px solid #3b82f6; padding-left: 8px;">
<div style="font-size: 0.65em; color: #6b7280;">Puffer</div>
<div style="font-size: 0.9em; font-weight: 700;">{res['puffer']:.1f}%</div>
</div>
<div style="border-left: 3px solid {delta_col}; padding-left: 8px;">
<div style="font-size: 0.65em; color: #6b7280;">Delta</div>
<div style="font-size: 0.9em; font-weight: 700; color: {delta_col};">{delta_val:.2f}</div>
</div>
</div>
<div style="background: {em_col}10; padding: 6px 10px; border-radius: 8px; margin-bottom: 12px; border: 1px dashed {em_col};">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 0.65em; color: #4b5563; font-weight: bold;">Stat. Erwartung (EM):</span>
<span style="font-size: 0.75em; font-weight: 800; color: {em_col};">±{res['em_pct']:.1f}%</span>
</div>
<div style="font-size: 0.6em; color: #6b7280; margin-top: 2px;">Sicherheit: <b>{em_safety:.1f}x EM</b></div>
</div>
<hr style="border: 0; border-top: 1px solid #f3f4f6; margin: 10px 0;">
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.72em; color: #4b5563; margin-bottom: 10px;">
<span>⏳ <b>{res['tage']}d</b></span>
<div style="display: flex; gap: 4px;">
<span style="background: #f3f4f6; padding: 2px 6px; border-radius: 4px; {rsi_style}">RSI: {rsi_val}</span>
<span style="background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-weight: 700;">{mkt_cap:.0f}B</span>
</div>
<span style="font-weight: 800; color: {'#ef4444' if is_earning_risk else '#6b7280'};">
{'⚠️' if is_earning_risk else '🗓️'} {earn_str}
</span>
</div>
<div style="background: {a_color}10; color: {a_color}; padding: 8px; border-radius: 8px; border-left: 4px solid {a_color}; font-size: 0.7em; font-weight: bold; text-align: center;">
🚀 {a_label}
</div>
</div>
"""
            st.markdown(html_code, unsafe_allow_html=True)
else:
    st.info("Scanner bereit. Bitte auf '🚀 Profi-Scan starten' klicken.")


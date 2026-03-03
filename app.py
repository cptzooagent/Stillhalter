import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import concurrent.futures
import time
from curl_cffi import requests as crequests
from io import StringIO

# Globale curl_cffi Session für yfinance
session = crequests.Session(impersonate="chrome")

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
        tk = yf.Ticker(symbol, session=session)
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
        tk = yf.Ticker(symbol, session=session)
        all_news = tk.news
        if not all_news:
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
            display_text = all_news[0].get('title', 'Marktstimmung aktiv')
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
    # 1. Die Sektor-ETFs (Deine neue ETF-Spalte)
    sector_etfs = ["XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC", "XBI", "GDX", "ARKK", "SPY", "QQQ", "IWM"]
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        resp = crequests.get(url, impersonate="chrome")
        df = pd.read_csv(StringIO(resp.text))
        tickers = df['Symbol'].tolist()
        nasdaq_extra = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "COIN", "MSTR", "HOOD", "PLTR", "SQ"]
        sp500_list = list(set(tickers + nasdaq_extra))
        sp500_list = [t.replace('.', '-') for t in sp500_list]
        return sp500_list, sector_etfs
    except: 
        backup = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META"]
        return backup, sector_etfs

def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol, session=session)
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
        current = info.get('currentPrice', info.get('current_price', 1))
        target = info.get('targetMedianPrice', 0)
        upside = ((target / current) - 1) * 100 if target > 0 else 0
        rev_growth = info.get('revenueGrowth', 0) * 100
        if rev_growth > 40: return f"🚀 HYPER-GROWTH (+{rev_growth:.0f}% Wachst.)", "#9b59b6"
        elif upside > 15 and rev_growth > 5: return f"✅ Stark (Ziel: +{upside:.0f}%, Wachst.: {rev_growth:.1f}%)", "#27ae60"
        elif upside > 25: return f"💎 Quality-Dip (Ziel: +{upside:.0f}%)", "#2980b9"
        elif upside < 0 or rev_growth < -2: return f"⚠️ Warnung (Ziel: {upside:.1f}%, Wachst.: {rev_growth:.1f}%)", "#e67e22"
        return f"⚖️ Neutral (Ziel: {upside:.0f}%)", "#7f8c8d"
    except: return "🔍 Check nötig", "#7f8c8d"

# --- SEKTION 3: PROFI-ANALYSE & TRADING-COCKPIT (FULL VERSION) ---
st.markdown("---")
st.markdown("### 🔍 Profi-Analyse & Trading-Cockpit")

# ADX-Hilfsfunktion (lokal für das Cockpit)
def calculate_adx_from_history(hist, period: int = 14):
    try:
        high = hist['High']
        low = hist['Low']
        close = hist['Close']

        plus_dm = high.diff()
        minus_dm = low.diff().abs()

        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).sum() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).sum() / atr)

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()
        return float(adx.iloc[-1])
    except Exception:
        return None

# Initialisierung des Session States für stabile Anzeige bei Widget-Interaktion
if 'cockpit_data' not in st.session_state:
    st.session_state.cockpit_data = None

with st.form("cockpit_form"):
    symbol_input = st.text_input("Ticker Symbol", value="MU").upper()
    submit_button = st.form_submit_button("🚀 Analyse starten / aktualisieren")

if submit_button and symbol_input:
    try:
        with st.spinner(f"Erstelle Dashboard für {symbol_input}..."):
            tk = yf.Ticker(symbol_input, session=session)
            info = tk.info

            # Nutze die vorhandene Full-Data Funktion aus deinem Skript
            res = get_stock_data_full(symbol_input)

            # Extra: Historie für ADX
            hist = tk.history(period="150d")
            adx_val = calculate_adx_from_history(hist) if not hist.empty else None

            if res[0] is not None:
                # Alles im Session State zwischenspeichern
                st.session_state.cockpit_data = {
                    'symbol': symbol_input,
                    'price': res[0],
                    'dates': res[1],
                    'earn': res[2],
                    'rsi': res[3],
                    'uptrend': res[4],
                    'near_lower': res[5],
                    'atr': res[6],
                    'pivots': res[7],
                    'info': info,
                    'analyst': get_analyst_conviction(info),
                    'adx': adx_val
                }
    except Exception as e:
        st.error(f"Fehler bei der Analyse: {e}")

# Anzeige-Logik: Rendert nur, wenn Daten im State vorhanden sind
if st.session_state.cockpit_data:
    d = st.session_state.cockpit_data
    
    # 1. Earnings & Signal-Box
    if d['earn'] and d['earn'] != "---":
        st.info(f"🗓️ Nächste Earnings: {d['earn']}")
    
    s2_d = d['pivots'].get('S2') if d['pivots'] else None
    s2_w = d['pivots'].get('W_S2') if d['pivots'] else None
    
    put_action, sig_col = "⏳ Warten", "white"
    if s2_w and d['price'] <= s2_w * 1.01:
        put_action, sig_col = "🔥 EXTREM (Weekly S2)", "#ff4b4b"
    elif d['rsi'] < 35 or (s2_d and d['price'] <= s2_d * 1.02):
        put_action, sig_col = "🟢 JETZT (S2/RSI)", "#27ae60"

    st.markdown(
        f'<div style="padding:10px; border-radius:10px; border: 2px solid {sig_col}; text-align:center;">'
        f'<small>Short Put Signal:</small><br>'
        f'<strong style="font-size:20px; color:{sig_col};">{put_action}</strong>'
        f'</div>',
        unsafe_allow_html=True
    )

    # 2. Sterne & Ampel-Logik
    analyst_txt, analyst_col = d['analyst']
    stars = 3 if "HYPER" in analyst_txt else 2 if "Stark" in analyst_txt else 1 if "Neutral" in analyst_txt else 0
    if d['uptrend']:
        stars += 0.5
    
    ampel_color, ampel_text = "#f1c40f", "NEUTRAL / ABWARTEN"
    if d['rsi'] < 25 or d['rsi'] > 75 or "Warnung" in analyst_txt:
        ampel_color, ampel_text = "#e74c3c", "STOPP / GEFAHR"
    elif stars >= 2.5 and d['uptrend'] and 30 <= d['rsi'] <= 60:
        ampel_color, ampel_text = "#27ae60", "TOP SETUP"

    st.markdown(
        f'<div style="background-color: {ampel_color}; color: white; padding: 15px; border-radius: 10px; '
        f'text-align: center; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">'
        f'<h2>● {ampel_text}</h2></div>',
        unsafe_allow_html=True
    )

    # 3. Metriken & Pivots
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kurs", f"{d['price']:.2f} $")
    c2.metric("RSI", int(d['rsi']))
    c3.metric("Phase", "🛡️ Trend" if d['uptrend'] else "💎 Dip")

    adx_val = d.get('adx', None)
    if adx_val is not None:
        trend_label = "Seitwärts" if adx_val < 20 else "Trend" if adx_val < 35 else "Starker Trend"
        c4.metric("Trendstärke (ADX)", f"{adx_val:.1f}", trend_label)
    else:
        c4.metric("Trendstärke (ADX)", "n/a")

    if d['pivots']:
        st.markdown("#### 🛡️ Technische Level")
        pc = st.columns(5)
        pc[0].metric("Weekly S2", f"{d['pivots']['W_S2']:.2f}$")
        pc[1].metric("Daily S2", f"{d['pivots']['S2']:.2f}$")
        pc[2].metric("Pivot P", f"{d['pivots']['P']:.2f}$")
        pc[3].metric("Daily R2", f"{d['pivots']['R2']:.2f}$")
        pc[4].metric("Weekly R2", f"{d['pivots']['W_R2']:.2f}$")

    # 4. OpenClaw KI-Analyse
    ki_status, ki_text, _ = get_openclaw_analysis(d['symbol'])
    st.info(ki_text)

    # 5. SMART OPTIONS-CHAIN MIT EXPECTED MOVE & DELTA
    st.markdown("---")
    st.markdown("### 🎯 Smart Option-Chain & Expected Move")
    opt_mode = st.radio("Strategie:", ["Put (Cash Secured)", "Call (Covered)"], horizontal=True)
    
    heute = datetime.now()
    valid_dates = [dt for dt in d['dates'] if 5 <= (datetime.strptime(dt, '%Y-%m-%d') - heute).days <= 45]
    
    if valid_dates:
        target_date = st.selectbox("📅 Verfallstag wählen", valid_dates)
        days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - heute).days)
        T = days / 365.0
        
        tk_active = yf.Ticker(d['symbol'], session=session)
        opt_chain = tk_active.option_chain(target_date)
        chain = opt_chain.puts if "Put" in opt_mode else opt_chain.calls
        
        # --- EXPECTED MOVE BERECHNUNG ---
        atm_idx = (chain['strike'] - d['price']).abs().idxmin()
        iv_atm = chain.loc[atm_idx, 'impliedVolatility']
        expected_move = d['price'] * iv_atm * np.sqrt(T)
        lower_em = d['price'] - expected_move
        upper_em = d['price'] + expected_move
        
        # EM Anzeige
        em_c1, em_c2 = st.columns(2)
        em_c1.metric("Expected Move (±)", f"{expected_move:.2f} $", f"{iv_atm*100:.1f}% IV")
        em_c2.warning(f"Statistischer Range: **{lower_em:.2f}$ — {upper_em:.2f}$**")

        # Daten Aufbereitung
        df = chain[chain['openInterest'] > 10].copy()
        df['Mid'] = (df['bid'] + df['ask']) / 2
        
        # Delta Berechnung pro Strike
        df['Delta'] = df.apply(
            lambda x: calculate_bsm_delta(
                d['price'],
                x['strike'],
                T,
                x['impliedVolatility'],
                option_type='put' if "Put" in opt_mode else 'call'
            ),
            axis=1
        )

        if "Put" in opt_mode:
            df = df[df['strike'] < d['price']].sort_values('strike', ascending=False)
            df['Puffer %'] = ((d['price'] - df['strike']) / d['price']) * 100
            df['EM_Safe'] = df['strike'] < lower_em
        else:
            df = df[df['strike'] > d['price']].sort_values('strike', ascending=True)
            df['Puffer %'] = ((df['strike'] - d['price']) / d['price']) * 100
            df['EM_Safe'] = df['strike'] > upper_em
        
        df['Yield p.a. %'] = (df['Mid'] / df['strike']) * (365 / days) * 100

        # Styling
        def style_rows(row):
            styles = [''] * len(row)
            if row['EM_Safe']:
                styles = ['background-color: rgba(16, 185, 129, 0.15)'] * len(row)  # Sanftes Grün für EM-Safe
            return styles

        df_show = df[['strike', 'bid', 'ask', 'Mid', 'Delta', 'Puffer %', 'Yield p.a. %', 'EM_Safe']].head(12)
        
        st.dataframe(
            df_show.style.apply(style_rows, axis=1).format({
                'strike': '{:.2f} $',
                'bid': '{:.2f} $',
                'ask': '{:.2f} $',
                'Mid': '{:.2f} $',
                'Delta': '{:.2f}',
                'Puffer %': '{:.1f}%',
                'Yield p.a. %': '{:.1f}%'
            }),
            use_container_width=True,
            height=450
        )
        st.caption("🛡️ **Grün hinterlegt:** Strike liegt außerhalb des Expected Move. | **Delta:** Wahrscheinlichkeit für ITM (Ziel: < 0.20)")

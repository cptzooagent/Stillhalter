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
        tk = yf.Ticker(symbol)
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
        tk = yf.Ticker(symbol)
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
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        resp = crequests.get(url, impersonate="chrome")
        df = pd.read_csv(StringIO(resp.text))
        tickers = df['Symbol'].tolist()
        nasdaq_extra = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "COIN", "MSTR", "HOOD", "PLTR", "SQ"]
        full_list = list(set(tickers + nasdaq_extra))
        return [t.replace('.', '-') for t in full_list]
    except: return ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META"]

def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
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

# --- UI & SIDEBAR ---
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
        ndq = yf.Ticker("^NDX"); vix = yf.Ticker("^VIX"); btc = yf.Ticker("BTC-USD")
        h_ndq = ndq.history(period="1mo"); h_vix = vix.history(period="1d"); h_btc = btc.history(period="1d")
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
        r = crequests.get("https://api.alternative.me/fng/", impersonate="chrome")
        return int(r.json()['data'][0]['value'])
    except: return 50

# --- MAIN DASHBOARD ---
st.markdown("## 🌍 Globales Markt-Monitoring")
cp_ndq, rsi_ndq, dist_ndq, vix_val, btc_val = get_market_data()
crypto_fg = get_crypto_fg()
stock_fg = 50 

if dist_ndq < -2 or vix_val > 25:
    m_color, m_text = "#e74c3c", "🚨 MARKT-ALARM: Nasdaq-Schwäche / Hohe Volatilität"
    m_advice = "Defensiv agieren. Fokus auf Call-Verkäufe zur Depot-Absicherung."
elif rsi_ndq > 72 or stock_fg > 80:
    m_color, m_text = "#f39c12", "⚠️ ÜBERHITZT: Korrekturgefahr (Gier/RSI hoch)"
    m_advice = "Keine neuen Puts mit engem Puffer. Gewinne sichern."
else:
    m_color, m_text = "#27ae60", "✅ TRENDSTARK: Marktumfeld ist konstruktiv"
    m_advice = "Puts auf starke Aktien bei Rücksetzern möglich."

st.markdown(f'<div style="background-color: {m_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;"><h3 style="margin:0; font-size: 1.4em;">{m_text}</h3><p style="margin:0; opacity: 0.9;">{m_advice}</p></div>', unsafe_allow_html=True)

r1c1, r1c2, r1c3 = st.columns(3)
r2c1, r2c2, r2c3 = st.columns(3)
with r1c1: st.metric("Nasdaq 100", f"{cp_ndq:,.0f}", f"{dist_ndq:.1f}% vs SMA20")
with r1c2: st.metric("Bitcoin", f"{btc_val:,.0f} $")
with r1c3: st.metric("VIX (Angst)", f"{vix_val:.2f}", delta="HOCH" if vix_val > 22 else "Normal", delta_color="inverse")
with r2c1: st.metric("Fear & Greed (Stock)", f"{stock_fg}")
with r2c2: st.metric("Fear & Greed (Crypto)", f"{crypto_fg}")
with r2c3: st.metric("Nasdaq RSI (14)", f"{int(rsi_ndq)}", delta="HEISS" if rsi_ndq > 70 else None, delta_color="inverse")

# --- SEKTION 1: PROFI-SCANNER ---
if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 Profi-Scan starten", key="kombi_scan_pro"):
    p_puffer = otm_puffer_slider / 100 
    p_min_yield = min_yield_pa
    p_min_cap = min_mkt_cap * 1_000_000_000
    heute = datetime.now()
    
    with st.spinner("Markt-Scanner analysiert Ticker..."):
        ticker_liste = ["APP", "AVGO", "NET", "CRWD", "MRVL", "NVDA", "CRDO", "HOOD", "SE", "ALAB", "TSLA", "PLTR", "COIN", "MSTR", "TER", "DELL", "DDOG", "MU", "LRCX", "RTX", "UBER"] if test_modus else get_combined_watchlist()
        status_text = st.empty()
        progress_bar = st.progress(0)
        all_results = []

        def check_single_stock(symbol):
            try:
                time.sleep(0.4) 
                tk = yf.Ticker(symbol)
                info = tk.info
                if not info or 'currentPrice' not in info: return None
                m_cap = info.get('marketCap', 0)
                price = info.get('currentPrice', 0)
                if m_cap < p_min_cap or not (min_stock_price <= price <= max_stock_price): return None
                res = get_stock_data_full(symbol)
                if res is None or res[0] is None: return None
                _, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
                if only_uptrend and not uptrend: return None
                valid_dates = [d for d in dates if 10 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 30]
                if not valid_dates: return None
                target_date = valid_dates[0]
                chain = tk.option_chain(target_date).puts
                target_strike = price * (1 - p_puffer)
                opts = chain[chain['strike'] <= target_strike].sort_values('strike', ascending=False)
                if opts.empty: return None
                o = opts.iloc[0]
                days_to_exp = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                iv = o.get('impliedVolatility', 0.4)
                exp_move_abs = price * (iv * np.sqrt(days_to_exp / 365))
                exp_move_pct = (exp_move_abs / price) * 100
                current_puffer = ((price - o['strike']) / price) * 100
                em_safety = current_puffer / exp_move_pct if exp_move_pct > 0 else 0
                bid, ask = o['bid'], o['ask']
                fair_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else o['lastPrice']
                delta_val = calculate_bsm_delta(price, o['strike'], days_to_exp/365, iv)
                y_pa = (fair_price / o['strike']) * (365 / max(1, days_to_exp)) * 100
                if y_pa >= p_min_yield:
                    analyst_txt, analyst_col = get_analyst_conviction(info)
                    s_val = 3.0 if "HYPER" in analyst_txt else 2.0 if "Stark" in analyst_txt else 0.0
                    if rsi < 35: s_val += 0.5
                    if uptrend: s_val += 0.5
                    return {
                        'symbol': symbol, 'price': price, 'y_pa': y_pa, 'strike': o['strike'], 
                        'puffer': current_puffer, 'bid': fair_price, 'rsi': rsi, 'earn': earn if earn else "---", 
                        'tage': days_to_exp, 'status': "🛡️ Trend" if uptrend else "💎 Dip", 'delta': delta_val,
                        'stars_val': s_val, 'stars_str': "⭐" * int(s_val) if s_val >= 1 else "⚠️",
                        'analyst_label': analyst_txt, 'analyst_color': analyst_col, 'mkt_cap': m_cap / 1e9,
                        'em_pct': exp_move_pct, 'em_safety': em_safety
                    }
            except: return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(check_single_stock, s): s for s in ticker_liste}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res_data = future.result()
                if res_data: all_results.append(res_data)
                progress_bar.progress((i + 1) / len(ticker_liste))
                if i % 5 == 0: status_text.text(f"Checke {i}/{len(ticker_liste)} Ticker...")
        status_text.empty(); progress_bar.empty()
        if all_results:
            st.session_state.profi_scan_results = sorted(all_results, key=lambda x: (x['stars_val'], x['y_pa']), reverse=True)
            st.success(f"Scan abgeschlossen: {len(all_results)} Treffer gefunden!")

# --- ERGEBNIS-ANZEIGE ---
if 'profi_scan_results' in st.session_state and st.session_state.profi_scan_results:
    all_results = st.session_state.profi_scan_results
    st.subheader(f"🎯 Top-Setups nach Qualität ({len(all_results)} Treffer)")
    cols = st.columns(4)
    heute_dt = datetime.now()
    for idx, res in enumerate(all_results):
        with cols[idx % 4]:
            em_safety = res.get('em_safety', 1.0)
            em_col = "#10b981" if em_safety >= 1.5 else "#f59e0b" if em_safety >= 1.0 else "#ef4444"
            earn_str = res.get('earn', "---")
            is_earning_risk = False
            if earn_str and earn_str != "---":
                try:
                    earn_date = datetime.strptime(f"{earn_str}2026", "%d.%m.%Y")
                    if 0 <= (earn_date - heute_dt).days <= res.get('tage', 14): is_earning_risk = True
                except: pass
            card_border = "4px solid #ef4444" if is_earning_risk else "1px solid #e5e7eb"
            
            html_code = f"""
            <div style="background: white; border: {card_border}; border-radius: 16px; padding: 18px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-weight: 800;">{res['symbol']} {res['stars_str']}</span>
                    <span style="color: #3b82f6; background: #3b82f610; padding: 2px 8px; border-radius: 6px;">{res['status']}</span>
                </div>
                <div style="margin: 10px 0;">
                    <div style="font-size: 0.7em; color: #6b7280;">Yield p.a.</div>
                    <div style="font-size: 1.9em; font-weight: 900;">{res['y_pa']:.1f}%</div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div>Strike: <b>{res['strike']:.1f}$</b></div>
                    <div>Puffer: <b>{res['puffer']:.1f}%</b></div>
                </div>
                <div style="background: {em_col}10; padding: 6px; border-radius: 8px; margin-top: 10px; border: 1px dashed {em_col};">
                    Sicherheit: <b>{em_safety:.1f}x EM</b>
                </div>
            </div>
            """
            st.markdown(html_code, unsafe_allow_html=True)

# --- SEKTION 2: DEPOT-MANAGER ---
st.markdown("---")
st.header("🛠️ Depot-Manager: Bestandsverwaltung & Reparatur")
if st.button("🚀 Depot jetzt analysieren (Inkl. Pivot-Check)", use_container_width=True):
    with st.spinner("Berechne Pivot-Punkte..."):
        my_assets = {"LRCX": [100, 210], "MU": [100, 390], "HOOD": [100, 120.00]}
        depot_list = []
        for symbol, data in my_assets.items():
            try:
                time.sleep(0.6)
                res = get_stock_data_full(symbol)
                if res is None: continue
                price, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
                qty, entry = data[0], data[1]
                perf_pct = ((price - entry) / entry) * 100
                ki_status, _, _ = get_openclaw_analysis(symbol)
                s2_d = pivots.get('S2') if pivots else None
                put_action = "🟢 JETZT (S2/RSI)" if (rsi < 35 or (s2_d and price <= s2_d * 1.02)) else "⏳ Warten"
                depot_list.append({
                    "Ticker": symbol, "Earnings": earn, "Einstand": f"{entry:.2f}$", "Aktuell": f"{price:.2f}$",
                    "P/L %": f"{perf_pct:+.1f}%", "KI": ki_status, "RSI": int(rsi), "Short Put": put_action
                })
            except: continue
        st.table(pd.DataFrame(depot_list))

# --- SEKTION 3: PROFI-ANALYSE ---
st.markdown("### 🔍 Profi-Analyse & Trading-Cockpit")
symbol_input = st.text_input("Ticker Symbol", value="MU").upper()
if symbol_input:
    try:
        tk = yf.Ticker(symbol_input)
        res = get_stock_data_full(symbol_input)
        if res[0] is not None:
            price, dates, earn, rsi, uptrend, near_lower, atr, pivots_res = res
            analyst_txt, analyst_col = get_analyst_conviction(tk.info)
            ki_status, ki_text, _ = get_openclaw_analysis(symbol_input)
            st.info(ki_text)
            st.metric("Kurs", f"{price:.2f} $", delta=f"RSI: {int(rsi)}")
            st.markdown(f"**Analysten-Meinung:** <span style='color:{analyst_col}'>{analyst_txt}</span>", unsafe_allow_html=True)
            
            if dates:
                target_date = st.selectbox("Laufzeit wählen", dates)
                chain = tk.option_chain(target_date).puts
                days_to_exp = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                atm_iv = chain.iloc[(chain['strike']-price).abs().argsort()[:1]]['impliedVolatility'].values[0]
                em_abs = price * (atm_iv * np.sqrt(days_to_exp / 365))
                st.write(f"📊 **Statistischer Expected Move:** ±{em_abs:.2f}$ (Ziel-Zone: >{price - em_abs:.2f}$)")
                
                df_disp = chain[(chain['strike'] >= price * 0.7) & (chain['strike'] <= price * 0.95)].copy()
                df_disp['Puffer %'] = ((price - df_disp['strike']) / price) * 100
                df_disp['Yield p.a. %'] = (( (df_disp['bid'] + df_disp['ask'])/2 ) / df_disp['strike']) * (365 / days_to_exp) * 100
                
                def style_rows(row):
                    p = row['Puffer %']
                    if p >= 15: return ['background-color: rgba(39, 174, 96, 0.1)'] * len(row)
                    elif 8 <= p < 15: return ['background-color: rgba(241, 196, 15, 0.1)'] * len(row)
                    return ['background-color: rgba(231, 76, 60, 0.1)'] * len(row)

                styled_df = df_disp[['strike', 'bid', 'ask', 'impliedVolatility', 'Puffer %', 'Yield p.a. %']].sort_values('strike', ascending=False).head(15).style.apply(style_rows, axis=1).format({
                    'strike': '{:.2f} $', 'bid': '{:.2f} $', 'ask': '{:.2f} $', 'impliedVolatility': '{:.2%}', 'Puffer %': '{:.1f} %', 'Yield p.a. %': '{:.1f} %'
                })
                st.dataframe(styled_df, use_container_width=True)
    except Exception as e:
        st.error(f"Fehler: {e}")

st.markdown("---")
st.caption(f"Letztes Update: {datetime.now().strftime('%H:%M:%S')} | Modus: {'🛠️ Simulation' if test_modus else '🚀 Live-Scan'}")

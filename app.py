import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 1. MATHE: DELTA-BERECHNUNG ---
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

# --- 2. DATEN-FUNKTIONEN ---
@st.cache_data(ttl=86400)
def get_combined_watchlist():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = df['Symbol'].tolist()
        nasdaq_extra = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "COIN", "MSTR", "HOOD", "PLTR", "SQ"]
        full_list = list(set(tickers + nasdaq_extra))
        return [t.replace('.', '-') for t in full_list]
    except Exception as e:
        return ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "COIN", "MSTR"]

@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period="150d") 
        if hist.empty: return None, [], "", 50, True, False, 0
            
        price = hist['Close'].iloc[-1] 
        dates = list(tk.options)
        
        # Indikatoren
        rsi_series = calculate_rsi(hist['Close'])
        rsi_val = rsi_series.iloc[-1]
        sma_200 = hist['Close'].mean() 
        is_uptrend = price > sma_200
        
        # Bollinger & ATR
        sma_20 = hist['Close'].rolling(window=20).mean()
        std_20 = hist['Close'].rolling(window=20).std()
        lower_band = (sma_20 - 2 * std_20).iloc[-1]
        is_near_lower = price <= (lower_band * 1.02)
        high_low = hist['High'] - hist['Low']
        atr = high_low.rolling(window=14).mean().iloc[-1]
            
        # Earnings-Check
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None:
                if isinstance(cal, dict) and 'Earnings Date' in cal:
                    earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
                elif hasattr(cal, 'columns') and 'Earnings Date' in cal.columns:
                    earn_str = cal['Earnings Date'].iloc[0].strftime('%d.%m.')
        except: pass
        
        return price, dates, earn_str, rsi_val, is_uptrend, is_near_lower, atr
    except:
        return None, [], "", 50, True, False, 0

# --- UI: SIDEBAR ---
# Das "with" Statement stellt sicher, dass alles links landet, 
# selbst wenn der Hauptcode weiter unten einen Fehler hat.
with st.sidebar:
    st.header("🛡️ Strategie-Einstellungen")
    otm_puffer_slider = st.slider("Gewünschter Puffer (%)", 3, 25, 10, key="puffer_sid")
    min_yield_pa = st.number_input("Mindestrendite p.a. (%)", 0, 100, 15, key="yield_sid")
    min_stock_price, max_stock_price = st.slider("Aktienpreis-Spanne ($)", 0, 1000, (20, 500), key="price_sid")

    st.markdown("---")
    only_uptrend = st.checkbox("Nur Aufwärtstrend (SMA 200)", value=False, key="trend_sid")
    st.info("Tipp: Deaktiviere den Aufwärtstrend für mehr Treffer am Wochenende.")

# --- DAS ULTIMATIVE MARKT-DASHBOARD (4 SPALTEN MIT DELTAS) ---
st.markdown("## 📊 Globales Marktwetter")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

# 1. Stock Fear & Greed (Proxy via SPY RSI)
try:
    spy_hist = yf.Ticker("SPY").history(period="30d")
    spy_rsi_series = calculate_rsi(spy_hist['Close'])
    rsi_val = spy_rsi_series.iloc[-1]
    rsi_prev = spy_rsi_series.iloc[-2]
    rsi_delta = rsi_val - rsi_prev
    fng_text = "Neutral" if 40 < rsi_val < 60 else "Greed" if rsi_val >= 60 else "Fear"
    m_col1.metric("Stock Sentiment (RSI)", f"{rsi_val:.0f}/100", f"{rsi_delta:+.1f} ({fng_text})")
except:
    m_col1.error("Stock F&G n.a.")

# 2. Crypto Fear & Greed (API)
try:
    import requests
    fg_data = requests.get("https://api.alternative.me/fng/").json()
    fg_crypto = int(fg_data['data'][0]['value'])
    fg_c_text = fg_data['data'][0]['value_classification']
    # Da die API kein Delta liefert, zeigen wir den Text als Delta-Ersatz
    m_col2.metric("Crypto Fear & Greed", f"{fg_crypto}/100", fg_c_text)
except:
    m_col2.error("Crypto F&G n.a.")

# 3. VIX (Angst-Index) mit Änderung
try:
    vix_data = yf.Ticker("^VIX").history(period="2d")
    vix_val = vix_data['Close'].iloc[-1]
    vix_prev = vix_data['Close'].iloc[-2]
    v_delta = vix_val - vix_prev
    # delta_color="inverse": VIX steigt = Rot, VIX sinkt = Grün
    m_col3.metric("VIX (Angst-Index)", f"{vix_val:.2f}", f"{v_delta:+.2f}", delta_color="inverse")
except:
    m_col3.error("VIX n.a.")

# 4. Bitcoin (Risk-On) mit Änderung
try:
    btc_data = yf.Ticker("BTC-USD").history(period="2d")
    btc_price = btc_data['Close'].iloc[-1]
    btc_prev = btc_data['Close'].iloc[-2]
    btc_delta = btc_price - btc_prev
    m_col4.metric("Bitcoin (Risk-On)", f"{btc_price:,.0f} $", f"{btc_delta:+.2f} $")
except:
    m_col4.error("BTC n.a.")

st.markdown("---")


# --- SEKTION 1: KOMBI-SCAN (FINALE EARNINGS-SPERRE) ---
st.markdown("---")
st.header("🔍 Kombi-Scan: Ultra-Safe (Harter Earnings-Stop)")

if st.button("🚀 Safe-Scan starten", key="kombi_scan_final_safe"):
    puffer_limit = otm_puffer_slider / 100 
    
    with st.spinner("Prüfe Termine... Trades werden streng vor Earnings begrenzt!"):
        ticker_liste = get_combined_watchlist()
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    all_results = []
    
    for i, symbol in enumerate(ticker_liste):
        progress_bar.progress((i + 1) / len(ticker_liste))
        status_text.text(f"Checke {symbol}...")
        
        try:
            res = get_stock_data_full(symbol)
            if res[0] is None: continue
            price, dates, earn, rsi, uptrend, near_lower, lower_band = res
            
            # --- DER HARTE FILTER ---
            max_days_allowed = 24 
            er_info_text = ""
            
            # WICHTIG: Wir prüfen, ob 'earn' ein Datum enthält (z.B. "11.02.")
            if earn and isinstance(earn, str) and "." in earn:
                try:
                    # Wir bauen das Datum korrekt zusammen (Tag.Monat.Jahr)
                    heute = datetime.now()
                    tag, monat = earn.split(".")[:2]
                    er_datum = datetime(heute.year, int(monat), int(tag))
                    
                    # Falls das Datum im Januar liegt, wir aber im Dezember sind -> nächstes Jahr
                    if er_datum < heute - timedelta(days=1):
                        er_datum = datetime(heute.year + 1, int(monat), int(tag))
                    
                    days_until_earn = (er_datum - heute).days
                    
                    # REGEL: Exit muss 2 Tage VOR ER sein
                    max_days_allowed = days_until_earn - 2
                    er_info_text = f"ER: {earn}"
                    
                    # Wenn weniger als 11 Tage Zeit bleiben bis zum Safe-Exit -> Ticker weg!
                    if max_days_allowed < 11:
                        continue 
                except:
                    er_info_text = "" # Falls Format-Fehler, keine Sperre (Fallback)

            # Suche Laufzeiten nur im "Safe-Fenster"
            valid_dates = [
                d for d in dates 
                if 11 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= max_days_allowed
            ]
            
            if not valid_dates: continue

            tk = yf.Ticker(symbol)
            best_opt = None
            max_y = -1
            
            for d_str in valid_dates:
                chain = tk.option_chain(d_str).puts
                m_strike = price * (1 - puffer_limit)
                opts = chain[chain['strike'] <= m_strike].sort_values('strike', ascending=False)
                
                if not opts.empty:
                    o = opts.iloc[0]
                    d_days = (datetime.strptime(d_str, '%Y-%m-%d') - heute).days
                    d_bid = o['bid'] if o['bid'] > 0 else o['lastPrice']
                    
                    if d_bid > (price * 0.04): continue # Split-Schutz
                    
                    curr_y = (d_bid / o['strike']) * (365 / max(1, d_days)) * 100
                    if curr_y > max_y:
                        max_y = curr_y
                        best_opt = {'o': o, 'days': d_days, 'date': d_str}

            if best_opt:
                # Rating
                score = 0
                if best_opt['o']['strike'] < lower_band: score += 1
                if 35 <= rsi <= 60: score += 1
                if uptrend: score += 1
                
                all_results.append({
                    'symbol': symbol, 'price': price, 'y_pa': max_y, 
                    'strike': best_opt['o']['strike'],
                    'puffer': ((price - best_opt['o']['strike']) / price) * 100,
                    'bid': best_opt['o']['bid'] if best_opt['o']['bid'] > 0 else best_opt['o']['lastPrice'],
                    'rsi': rsi, 'earn': er_info_text, 'tage': best_opt['days'], 
                    'score': score, 'stars': "⭐" * score if score > 0 else "⚪"
                })
        except: continue

    # --- ANZEIGE ---
    status_text.empty()
    progress_bar.empty()

    if not all_results:
        st.warning("Keine sicheren Trades vor Earnings gefunden.")
    else:
        all_results = sorted(all_results, key=lambda x: (x['score'], x['y_pa']), reverse=True)
        st.success(f"Scan fertig: {len(all_results)} 'Safe-Exit' Trades gefunden!")
        
        cols = st.columns(4)
        for idx, res in enumerate(all_results):
            with cols[idx % 4]:
                # Visualisierung
                safe_label = f"<div style='color:#27ae60; font-weight:bold; font-size:0.75em;'>🛡️ Exit vor {res['earn']}</div>" if res['earn'] else "<div style='color:#7f8c8d; font-size:0.75em;'>Keine ER im Zeitfenster</div>"
                rsi_c = "#e74c3c" if res['rsi'] > 70 else "#2ecc71" if res['rsi'] < 40 else "#555"
                
                with st.container(border=True):
                    st.markdown(f"**{res['symbol']}** {res['stars']}{safe_label}", unsafe_allow_html=True)
                    st.metric("Yield p.a.", f"{res['y_pa']:.1f}%")
                    st.markdown(f"""
                    <div style="font-size: 0.85em; background:#f0fff0; padding:10px; border-radius:8px; border-left:5px solid #27ae60;">
                    Kurs: <b>{res['price']:.2f}$</b> | Strike: <b>{res['strike']:.1f}$</b><br>
                    <hr style="margin:5px 0; border-top:1px solid #c8e6c9;">
                    <b>Laufzeit: {res['tage']} Tage (Safe)</b><br>
                    Puffer: {res['puffer']:.1f}% | Bid: {res['bid']:.2f}$<br>
                    RSI: <span style="color:{rsi_c}; font-weight:bold;">{res['rsi']:.0f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
# --- SEKTION 2: SMART DEPOT-MANAGER (REPAIR VERSION) ---
st.markdown("### 💼 Smart Depot-Manager (Aktiv)")
depot_data = [
    {'Ticker': 'AFRM', 'Einstand': 76.00}, {'Ticker': 'HOOD', 'Einstand': 120.0},
    {'Ticker': 'JKS', 'Einstand': 50.00}, {'Ticker': 'GTM', 'Einstand': 17.00},
    {'Ticker': 'HIMS', 'Einstand': 37.00}, {'Ticker': 'NVO', 'Einstand': 97.00},
    {'Ticker': 'RBRK', 'Einstand': 70.00}, {'Ticker': 'SE', 'Einstand': 170.00},
    {'Ticker': 'ETSY', 'Einstand': 67.00}, {'Ticker': 'TTD', 'Einstand': 102.00},
    {'Ticker': 'ELF', 'Einstand': 109.00}
]       

p_cols = st.columns(4) 
for i, item in enumerate(depot_data):
    price, _, earn, rsi, uptrend, _, _ = get_stock_data_full(item['Ticker'])
    if price:
        diff = (price / item['Einstand'] - 1) * 100
        perf_color = "#2ecc71" if diff >= 0 else "#e74c3c"
        with p_cols[i % 4]:
            with st.container(border=True):
                st.markdown(f"**{item['Ticker']}** <span style='float:right; color:{perf_color}; font-weight:bold;'>{diff:+.1f}%</span>", unsafe_allow_html=True)
                
                # Strategie-Logik
                if diff < -20:
                    # Repair-Ansatz: Call 15% über aktuellem Kurs, egal wo der Einstand ist
                    repair_strike = price * 1.15
                    st.warning("🛠️ Repair-Modus")
                    st.caption(f"Call @{repair_strike:.1f}$ senkt Einstand.")
                    if rsi < 40: 
                        st.info("Wait: RSI zu tief")
                    else:
                        st.success(f"Prämie einsammeln!")
                elif rsi > 65:
                    st.success("🟢 Call-Chance!")
                else:
                    st.info("⏳ Seitwärts")
                
                if earn: st.warning(f"📅 ER: {earn}")

# --- SEKTION 3: EINZEL-CHECK (STABILE AMPEL-VERSION) ---
st.markdown("---")
st.subheader("🔍 Einzel-Check & Option-Chain")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: t_in = st.text_input("Ticker Symbol", value="HOOD").upper()

if t_in:
    ticker_symbol = t_in.strip().upper()
    with st.spinner(f"Analysiere {ticker_symbol}..."):
        price, dates, earn, rsi, uptrend, near_lower, atr = get_stock_data_full(ticker_symbol)
    
    if price is None:
        st.error(f"❌ Keine Daten für '{ticker_symbol}' gefunden.")
    elif not dates:
        st.warning(f"⚠️ Keine Optionen für {ticker_symbol} verfügbar.")
    else:
        h1, h2, h3 = st.columns(3)
        h1.metric("Kurs", f"{price:.2f}$")
        h2.metric("Trend", "📈 Bullisch" if uptrend else "📉 Bärisch")
        h3.metric("RSI", f"{rsi:.0f}")

        if not uptrend: st.error("🛑 Achtung: Aktie notiert unter SMA 200!")
        
        d_sel = st.selectbox("Laufzeit wählen", dates)
        try:
            tk = yf.Ticker(ticker_symbol)
            chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
            expiry_dt = datetime.strptime(d_sel, '%Y-%m-%d')
            days_to_expiry = max(1, (expiry_dt - datetime.now()).days)
            T = days_to_expiry / 365
            
            chain['delta_calc'] = chain.apply(lambda opt: calculate_bsm_delta(
                price, opt['strike'], T, (opt['impliedVolatility'] if opt['impliedVolatility'] else 0.4), option_type=mode
            ), axis=1)

            # --- VERBESSERTE FILTER-LOGIK (NUR OTM) ---
            if mode == "put":
                # Nur Strikes unterhalb des aktuellen Kurses (max. 98% vom Preis)
                filtered_df = chain[chain['strike'] <= price * 0.98].sort_values('strike', ascending=False)
            else:
                # Nur Strikes oberhalb des aktuellen Kurses (min. 102% vom Preis)
                filtered_df = chain[chain['strike'] >= price * 1.02].sort_values('strike', ascending=True)
            
            st.write("---")
            # Falls keine Optionen im OTM-Bereich gefunden wurden
            if filtered_df.empty:
                st.warning(f"Keine attraktiven OTM {mode.upper()}s im gewählten Bereich gefunden.")
            else:
                for _, opt in filtered_df.head(15).iterrows():
                    bid_val = opt['bid'] if not pd.isna(opt['bid']) and opt['bid'] > 0 else 0.05
                    d_abs = abs(opt['delta_calc'])
                    
                    # Ampelsystem nach Delta (Konservativ)
                    risk_emoji = "🟢" if d_abs < 0.15 else "🟡" if d_abs <= 0.25 else "🔴"
                    
                    y_pa = (bid_val / opt['strike']) * (365 / days_to_expiry) * 100
                    puffer = (abs(opt['strike'] - price) / price) * 100
                    
                    # Anzeige
                    st.markdown(
                        f"{risk_emoji} **Strike: {opt['strike']:.1f}** | "
                        f"Bid: **{bid_val:.2f}$** | "
                        f"Delta: {d_abs:.2f} | "
                        f"Puffer: {puffer:.1f}% | "
                        f"Yield: {y_pa:.1f}% p.a.",
                        unsafe_allow_html=True
                    )
        except Exception as e:
            st.error(f"Fehler bei der Anzeige: {e}")



















import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import concurrent.futures

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
    """Berechnet Hybrid-Pivots: Klassisch + Fibonacci (Daily & Weekly)."""
    try:
        tk = yf.Ticker(symbol)
        # Daily Daten
        hist_d = tk.history(period="5d") 
        if len(hist_d) < 2: return None
        l_day = hist_d.iloc[-2]
        h, l, c = l_day['High'], l_day['Low'], l_day['Close']
        range_d = h - l
        p_d = (h + l + c) / 3
        
        # Weekly Daten (Brandmauer)
        hist_w = tk.history(period="3wk", interval="1wk")
        if len(hist_w) < 2: return None
        l_week = hist_w.iloc[-2]
        hw, lw, cw = l_week['High'], l_week['Low'], l_week['Close']
        range_w = hw - lw
        p_w = (hw + lw + cw) / 3

        return {
            "P": p_d, 
            "Std_S1": (2 * p_d) - h,        # Klassischer S1
            "Std_S2": p_d - (h - l),       # Klassischer S2
            "Fib_S2": p_d - (range_d * 0.618), # Fib 61.8% (Golden Ratio)
            "W_Fib_S2": p_w - (range_w * 0.618) # Weekly Golden Ratio
        }
    except: return None
        
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
    except:
        return ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META"]

@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period="150d") 
        if hist.empty: return None, [], "", 50, True, False, 0, None
        
        price = hist['Close'].iloc[-1] 
        dates = list(tk.options)
        rsi_series = calculate_rsi(hist['Close'])
        rsi_val = rsi_series.iloc[-1]
        
        # Trend-Check (SMA 200)
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else hist['Close'].mean()
        is_uptrend = price > sma_200
        
        # Bollinger-Band Check
        sma_20 = hist['Close'].rolling(window=20).mean()
        std_20 = hist['Close'].rolling(window=20).std()
        lower_band = (sma_20 - 2 * std_20).iloc[-1]
        is_near_lower = price <= (lower_band * 1.02)
        
        atr = (hist['High'] - hist['Low']).rolling(window=14).mean().iloc[-1]
        
        # NEU: Pivots hier zentral abrufen
        pivots = calculate_pivots(symbol)
        
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        
        # Rückgabe von 8 Werten (pivots ist der letzte)
        return price, dates, earn_str, rsi_val, is_uptrend, is_near_lower, atr, pivots
    except:
        return None, [], "", 50, True, False, 0, None
        
# --- UI: SIDEBAR (KOMPLETT-REPARATUR) ---
with st.sidebar:
    st.header("🛡️ Strategie-Einstellungen")
    
    # Basis-Filter
    otm_puffer_slider = st.slider("Gewünschter Puffer (%)", 3, 25, 10, key="puffer_sid")
    min_yield_pa = st.number_input("Mindestrendite p.a. (%)", 0, 100, 15, key="yield_sid")
    
    # Der Aktienpreis-Regler
    min_stock_price, max_stock_price = st.slider(
        "Aktienpreis-Spanne ($)", 
        0, 1000, (20, 500), 
        key="price_sid"
    )

    st.markdown("---")
    st.subheader("Qualitäts-Filter")
    
    # Marktkapitalisierung & Trend
    min_mkt_cap = st.slider("Mindest-Marktkapitalisierung (Mrd. $)", 1, 1000, 50, key="mkt_cap_sid")
    only_uptrend = st.checkbox("Nur Aufwärtstrend (SMA 200)", value=False, key="trend_sid")
    
    # WICHTIG: Die Checkbox für den Simulationsmodus
    test_modus = st.checkbox("🛠️ Simulations-Modus (Test)", value=False, key="sim_checkbox")
    
    st.markdown("---")
    st.info("💡 Profi-Tipp: Für den S&P 500 Scan ab 16:00 Uhr 'Simulations-Modus' deaktivieren.")
    
# --- HILFSFUNKTIONEN FÜR DAS DASHBOARD ---

def get_market_data():
    """Holt globale Marktdaten mit korrektem Nasdaq 100 Index (^NDX)."""
    try:
        ndq = yf.Ticker("^NDX")
        vix = yf.Ticker("^VIX")
        btc = yf.Ticker("BTC-USD")
        
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
    except:
        return 0, 50, 0, 20, 0

def get_crypto_fg():
    try:
        import requests
        r = requests.get("https://api.alternative.me/fng/")
        return int(r.json()['data'][0]['value'])
    except:
        return 50

# --- HAUPT-BLOCK: GLOBAL MONITORING ---

st.markdown("## 🌍 Globales Markt-Monitoring")

# Daten abrufen
cp_ndq, rsi_ndq, dist_ndq, vix_val, btc_val = get_market_data()
crypto_fg = get_crypto_fg()
# Hinweis: fg_val (Stock) müsstest du aus deiner vorhandenen Funktion nehmen
stock_fg = 50 # Platzhalter, falls deine Funktion anders heißt

# Master-Banner Logik
if dist_ndq < -2 or vix_val > 25:
    m_color, m_text = "#e74c3c", "🚨 MARKT-ALARM: Nasdaq-Schwäche / Hohe Volatilität"
    m_advice = "Defensiv agieren. Fokus auf Call-Verkäufe zur Depot-Absicherung."
elif rsi_ndq > 72 or stock_fg > 80:
    m_color, m_text = "#f39c12", "⚠️ ÜBERHITZT: Korrekturgefahr (Gier/RSI hoch)"
    m_advice = "Keine neuen Puts mit engem Puffer. Gewinne sichern."
else:
    m_color, m_text = "#27ae60", "✅ TRENDSTARK: Marktumfeld ist konstruktiv"
    m_advice = "Puts auf starke Aktien bei Rücksetzern möglich."

st.markdown(f"""
    <div style="background-color: {m_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h3 style="margin:0; font-size: 1.4em;">{m_text}</h3>
        <p style="margin:0; opacity: 0.9;">{m_advice}</p>
    </div>
""", unsafe_allow_html=True)

# Das 2x3 Raster
r1c1, r1c2, r1c3 = st.columns(3)
r2c1, r2c2, r2c3 = st.columns(3)

with r1c1:
    st.metric("Nasdaq 100", f"{cp_ndq:,.0f}", f"{dist_ndq:.1f}% vs SMA20")
with r1c2:
    st.metric("Bitcoin", f"{btc_val:,.0f} $")
with r1c3:
    st.metric("VIX (Angst)", f"{vix_val:.2f}", delta="HOCH" if vix_val > 22 else "Normal", delta_color="inverse")

with r2c1:
    st.metric("Fear & Greed (Stock)", f"{stock_fg}")
with r2c2:
    st.metric("Fear & Greed (Crypto)", f"{crypto_fg}")
with r2c3:
    st.metric("Nasdaq RSI (14)", f"{int(rsi_ndq)}", delta="HEISS" if rsi_ndq > 70 else None, delta_color="inverse")

st.markdown("---")


# --- NEUE ANALYSTEN-LOGIK (VOR DEM SCAN DEFINIEREN) ---
def get_analyst_conviction(info):
    try:
        current = info.get('current_price', info.get('currentPrice', 1))
        target = info.get('targetMedianPrice', 0)
        upside = ((target / current) - 1) * 100 if target > 0 else 0
        rev_growth = info.get('revenueGrowth', 0) * 100
        
        # 1. LILA: 🚀 HYPER-GROWTH (APP, CRDO, NVDA, ALAB)
        if rev_growth > 40:
            return f"🚀 HYPER-GROWTH (+{rev_growth:.0f}% Wachst.)", "#9b59b6"
        # 2. GRÜN: ✅ STARK (AVGO, CRWD)
        elif upside > 15 and rev_growth > 5:
            return f"✅ Stark (Ziel: +{upside:.0f}%, Wachst.: {rev_growth:.1f}%)", "#27ae60"
        # 3. BLAU: 💎 QUALITY-DIP (NET, MRVL)
        elif upside > 25:
            return f"💎 Quality-Dip (Ziel: +{upside:.0f}%)", "#2980b9"
        # 4. ORANGE: ⚠️ WARNUNG (GTM, stagnierende Werte)
        elif upside < 0 or rev_growth < -2:
            return f"⚠️ Warnung (Ziel: {upside:.1f}%, Wachst.: {rev_growth:.1f}%)", "#e67e22"
        return f"⚖️ Neutral (Ziel: {upside:.0f}%)", "#7f8c8d"
    except:
        return "🔍 Check nötig", "#7f8c8d"



# --- SEKTION: PROFI-SCANNER (HIGH-SPEED MULTITHREADING EDITION) ---
if st.button("🚀 Profi-Scan starten (High Speed)", key="kombi_scan_pro"):
    puffer_limit = otm_puffer_slider / 100 
    mkt_cap_limit = min_mkt_cap * 1_000_000_000
    heute = datetime.now()
    
    with st.spinner("Markt-Scanner läuft auf Hochtouren..."):
        if test_modus:
            ticker_liste = ["APP", "AVGO", "NET", "CRWD", "MRVL", "NVDA", "CRDO", "HOOD", "SE", "ALAB", "TSLA", "PLTR", "COIN", "MSTR", "TER", "DELL", "DDOG", "MU", "LRCX", "RTX", "UBER"]
        else:
            ticker_liste = get_combined_watchlist()
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    all_results = []

    # --- DIE UNTER-FUNKTION FÜR DEN PARALLELEN CHECK ---
    def check_single_stock(symbol):
        try:
            tk = yf.Ticker(symbol)
            info = tk.info
            curr_price = info.get('currentPrice', 0)
            m_cap = info.get('marketCap', 0)
            
            # Filter 1: Basis-Daten (Preis & Market Cap)
            if m_cap < mkt_cap_limit: return None
            if not (min_stock_price <= curr_price <= max_stock_price): return None
            
            # Daten abrufen (8 Werte!)
            res = get_stock_data_full(symbol)
            if res[0] is None: return None
            price, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
            
            # Filter 2: Trend-Check
            if only_uptrend and not uptrend: return None

            # Filter 3: Laufzeit & Earnings-Schutz
            max_days_allowed = 24
            if earn and "." in earn:
                try:
                    tag, monat = earn.split(".")[:2]
                    er_datum = datetime(heute.year, int(monat), int(tag))
                    if er_datum < heute: er_datum = datetime(heute.year + 1, int(monat), int(tag))
                    max_days_allowed = min(24, (er_datum - heute).days - 2)
                except: pass

            # Verfallstage filtern (11 bis max_days_allowed)
            valid_dates = [d for d in dates if 11 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= max_days_allowed]
            if not valid_dates: return None
            
            # Options-Analyse (Target Strike unter Puffer)
            target_date = valid_dates[0]
            chain = tk.option_chain(target_date).puts
            target_strike = price * (1 - puffer_limit)
            opts = chain[chain['strike'] <= target_strike].sort_values('strike', ascending=False)
            
            if not opts.empty:
                o = opts.iloc[0]
                bid_val = o['bid'] if o['bid'] > 0 else o['lastPrice']
                days = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                y_pa = (bid_val / o['strike']) * (365 / max(1, days)) * 100
                
                # Filter 4: Rendite-Check
                if y_pa >= min_yield_pa:
                    analyst_txt, analyst_col = get_analyst_conviction(info)
                    
                    # Sterne-Logik (Qualitäts-Score)
                    stars = 0
                    if "HYPER" in analyst_txt: stars = 3
                    elif "Stark" in analyst_txt: stars = 2
                    elif "Neutral" in analyst_txt: stars = 1
                    
                    if rsi < 30: stars -= 1 # Technisches Risiko
                    if rsi > 75: stars -= 0.5 # Überhitzt
                    if uptrend and stars > 0: stars += 0.5 
                    stars = max(0, float(stars))

                    return {
                        'symbol': symbol, 'price': price, 'y_pa': y_pa, 
                        'strike': o['strike'], 'puffer': ((price - o['strike']) / price) * 100,
                        'bid': bid_val, 'rsi': rsi, 'earn': earn if earn else "n.a.", 
                        'tage': days, 'status': "🛡️ Trend" if uptrend else "💎 Dip",
                        'stars_val': stars, 'stars_str': "⭐" * int(stars) if stars >= 1 else "⚠️",
                        'analyst_txt': analyst_txt, 'analyst_col': analyst_col,
                        'mkt_cap': m_cap / 1_000_000_000
                    }
        except: return None
        return None

    # --- EXECUTION: 15 THREADS GLEICHZEITIG ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(check_single_stock, s): s for s in ticker_liste}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res_data = future.result()
            if res_data:
                all_results.append(res_data)
            
            # UI Update
            progress_bar.progress((i + 1) / len(ticker_liste))
            if i % 5 == 0:
                status_text.text(f"Analysiere {i}/{len(ticker_liste)}: {future_to_symbol[future] if 'future_to_symbol' in locals() else 'Ticker'}...")

    status_text.empty()
    progress_bar.empty()

    # --- RESULTATE ANZEIGEN ---
    if not all_results:
        st.warning("Keine Treffer gefunden, die den Kriterien entsprechen.")
    else:
        # Sortierung: Qualität (Sterne) -> Rendite
        all_results = sorted(all_results, key=lambda x: (x['stars_val'], x['y_pa']), reverse=True)

        st.markdown(f"### 🎯 Top-Setups ({len(all_results)} Treffer)")
        cols = st.columns(4)
        for idx, res in enumerate(all_results):
            with cols[idx % 4]:
                s_color = "#27ae60" if "🛡️" in res['status'] else "#2980b9"
                border_color = res['analyst_col'] if res['stars_val'] >= 2 else "#e0e0e0"
                rsi_col = "#e74c3c" if res['rsi'] > 70 or res['rsi'] < 30 else "#7f8c8d"
                
                with st.container(border=True):
                    st.markdown(f"**{res['symbol']}** {res['stars_str']} <span style='float:right; font-size:0.75em; color:{s_color}; font-weight:bold;'>{res['status']}</span>", unsafe_allow_html=True)
                    st.metric("Yield p.a.", f"{res['y_pa']:.1f}%")
                    
                    st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 8px; border-radius: 5px; border: 2px solid {border_color}; margin-bottom: 8px; font-size: 0.85em;">
                            🎯 Strike: <b>{res['strike']:.1f}$</b> | 💰 Bid: <b>{res['bid']:.2f}$</b><br>
                            🛡️ Puffer: <b>{res['puffer']:.1f}%</b> | ⏳ Tage: <b>{res['tage']}</b>
                        </div>
                        <div style="font-size: 0.8em; color: #7f8c8d; margin-bottom: 5px;">
                            📅 ER: <b>{res['earn']}</b> | RSI: <b style="color:{rsi_col};">{int(res['rsi'])}</b>
                        </div>
                        <div style="font-size: 0.85em; border-left: 4px solid {res['analyst_col']}; padding: 4px 8px; font-weight: bold; color: {res['analyst_col']}; background: {res['analyst_col']}10; border-radius: 0 4px 4px 0;">
                            {res['analyst_txt']}
                        </div>
                    """, unsafe_allow_html=True)
                    
# --- SEKTION 4: DEPOT-MANAGER (DEIN ECHTES DEPOT) ---
st.markdown("---")
st.header("🛠️ Depot-Manager: Bestandsverwaltung & Reparatur")

my_assets = {
    "AFRM": [100, 76.00], "ELF": [100, 109.00], "ETSY": [100, 67.00],
    "GTLB": [100, 41.00], "GTM": [100, 17.00], "HIMS": [100, 36.00],
    "HOOD": [100, 120.00], "JKS": [100, 50.00], "NVO": [100, 97.00],
    "RBRK": [100, 70.00], "SE": [100, 170.00], "TTD": [100, 102.00]
}

with st.expander("📂 Mein Depot & Strategie-Signale", expanded=True):
    depot_list = []
    for symbol, data in my_assets.items():
        try:
            res = get_stock_data_full(symbol)
            if res[0] is None: continue
            
            # Hier entpacken wir jetzt 8 Werte
            price, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
            qty, entry = data[0], data[1]
            perf_pct = ((price - entry) / entry) * 100
            
            s2_d = pivots['S2'] if pivots else 0
            s2_w = pivots['W_S2'] if pivots else 0
            
            # Reparatur-Logik
            put_action = "🟢 JETZT (S2/RSI)" if (rsi < 35 or price <= s2_d * 1.02) else "⏳ Warten"
            if price <= s2_w * 1.01:
                put_action = "🔥 EXTREM (Weekly S2)"
            
            call_action = "🟢 JETZT (RSI > 55)" if rsi > 55 else "⏳ Warten"

            depot_list.append({
                "Ticker": symbol,
                "Einstand": f"{entry:.2f} $",
                "Aktuell": f"{price:.2f} $",
                "P/L %": f"{perf_pct:+.1f}%",
                "RSI": int(rsi),
                "Short Put (Repair)": put_action,
                "Covered Call": call_action,
                "S2 Daily": f"{s2_d:.2f} $",
                "S2 Weekly": f"{s2_w:.2f} $"
            })
        except: continue
    
    if depot_list:
        st.table(pd.DataFrame(depot_list))
        
    st.info("💡 **Strategie:** Wenn 'Short Put' auf 🔥 steht, ist die Aktie am wöchentlichen Tiefstand – technisch das sicherste Level zum Verbilligen.")

                    
# --- SEKTION 3: DESIGN-UPGRADE & SICHERHEITS-AMPEL (INKL. PANIK-SCHUTZ) ---
st.markdown("### 🔍 Profi-Analyse & Trading-Cockpit")
symbol_input = st.text_input("Ticker Symbol", value="MU", help="Gib ein Ticker-Symbol ein").upper()

if symbol_input:
    try:
        with st.spinner(f"Erstelle Dashboard für {symbol_input}..."):
            tk = yf.Ticker(symbol_input)
            info = tk.info
            res = get_stock_data_full(symbol_input)
            
            # Ändere diese Zeile in Sektion 3:
            if res[0] is not None:
                price, dates, earn, rsi, uptrend, near_lower, atr, pivots_res = res  # pivots_res hinzugefügt
                analyst_txt, analyst_col = get_analyst_conviction(info)
                
                # Sterne-Logik (Basis für Qualität)
                stars = 0
                if "HYPER" in analyst_txt: stars = 3
                elif "Stark" in analyst_txt: stars = 2
                elif "Neutral" in analyst_txt: stars = 1
                if uptrend and stars > 0: stars += 0.5
                
                # --- VERSCHÄRFTE AMPEL-LOGIK (PANIK-SCHUTZ) ---
                ampel_color, ampel_text = "#f1c40f", "NEUTRAL / ABWARTEN"
                
                if rsi < 25:
                    # Panik-Schutz greift zuerst
                    ampel_color, ampel_text = "#e74c3c", "STOPP: PANIK-ABVERKAUF (RSI < 25)"
                elif rsi > 75:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: ÜBERHITZT (RSI > 75)"
                elif stars >= 2.5 and uptrend and 30 <= rsi <= 60:
                    # Ideales Setup
                    ampel_color, ampel_text = "#27ae60", "TOP SETUP (Sicher)"
                elif "Warnung" in analyst_txt:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: ANALYSTEN-WARNUNG"
                else:
                    ampel_color, ampel_text = "#f1c40f", "NEUTRAL / ABWARTEN"

                # 1. HEADER: Ampel
                st.markdown(f"""
                    <div style="background-color: {ampel_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h2 style="margin:0; font-size: 1.8em; letter-spacing: 1px;">● {ampel_text}</h2>
                    </div>
                """, unsafe_allow_html=True)

                # 2. METRIKEN-BOARD
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Kurs", f"{price:.2f} $")
                with col2:
                    st.metric("RSI (14)", f"{int(rsi)}", delta="PANIK" if rsi < 25 else None, delta_color="inverse")
                with col3:
                    status_icon = "🛡️" if uptrend else "💎"
                    st.metric("Phase", f"{status_icon} {'Trend' if uptrend else 'Dip'}")
                with col4:
                    st.metric("Qualität", "⭐" * int(stars))

                # --- FIBONACCI-ZONEN ANZEIGE ---
                if pivots_res: # Nutzt die Rückgabe aus deinem get_stock_data_full
                    st.markdown("#### 🛡️ Fibonacci Support Level (Brandmauern)")
            
                    fz1, fz2, fz3 = st.columns(3)
            
                    fz1.metric("Fib 38.2% (Soft)", f"{pivots_res['S1']:.2f} $")
                    fz2.metric("Fib 61.8% (Stark)", f"{pivots_res['S2']:.2f} $", help="Golden Ratio: Hier kaufen oft Institutionen")
                    fz3.metric("Weekly Fib (Boden)", f"{pivots_res['W_S2']:.2f} $", delta="Max Safe", delta_color="inverse")
            
                    st.info(f"💡 **Strategie:** Für 'Keine Einbuchung' wähle einen Strike unter **{pivots_res['W_S2']:.2f} $**.")


                # 3. ANALYSTEN BOX
                st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 10px solid {analyst_col}; margin-top: 10px;">
                        <h4 style="margin-top:0; color: #31333F;">💡 Fundamentale Analyse</h4>
                        <p style="font-size: 1.1em; font-weight: bold; color: {analyst_col};">{analyst_txt}</p>
                        <hr style="margin: 10px 0;">
                        <span style="color: #555;">📅 Nächste Earnings: <b>{earn if earn else 'n.a.'}</b></span>
                    </div>
                """, unsafe_allow_html=True)

                # --- OPTION-CHAIN MIT TIEFEN-PUFFER-LOGIK ---
                st.markdown("### 🎯 Option-Chain Auswahl (Sicherheits-Fokus)")
                if dates:
                    target_date = st.selectbox("📅 Verfallstag wählen", dates[:5])
                    chain = tk.option_chain(target_date).puts
            
                    # NEU: Wir filtern hier bewusst auf Strikes, die mind. 8% unter dem Kurs liegen
                    min_puffer_filter = 0.08 
                    df_puts = chain[chain['strike'] < (price * (1 - min_puffer_filter))].sort_values('strike', ascending=False).head(15).copy()
            
                    # Falls die Liste leer ist (weil Puffer zu extrem), zeigen wir die nächsten verfügbaren an
                    if df_puts.empty:
                        df_puts = chain[chain['strike'] < price].sort_values('strike', ascending=False).head(12).copy()

                    def check_hybrid_safety(strike):
                        if strike < pivots_res['W_Fib_S2']: return "💎 MAX SAFETY (Weekly)"
                        if strike < min(pivots_res['Std_S2'], pivots_res['Fib_S2']): return "🟢 DOUBLE SUPPORT"
                        if strike < pivots_res['Fib_S2'] or strike < pivots_res['Std_S2']: return "🟡 SINGLE SUPPORT"
                        return "🔴 AGGRESSIV"

                    df_puts['Sicherheit'] = df_puts['strike'].apply(check_hybrid_safety)
                    df_puts['Puffer %'] = ((price - df_puts['strike']) / price) * 100
            
                    # Styling bleibt gleich
                    def color_hybrid(val):
                        colors = {"💎 MAX SAFETY (Weekly)": "#1a5276", "🟢 DOUBLE SUPPORT": "#1e8449", 
                                  "🟡 SINGLE SUPPORT": "#d4ac0d", "🔴 AGGRESSIV": "#922b21"}
                        return f'background-color: {colors.get(val, "white")}; color: white; font-weight: bold'

                    styled_df = df_puts[['strike', 'bid', 'Puffer %', 'Sicherheit']].style.applymap(color_hybrid, subset=['Sicherheit']).format({
                        'strike': '{:.2f} $', 'bid': '{:.2f} $', 'Puffer %': '{:.1f} %'
                    })

                    st.dataframe(styled_df, use_container_width=True)

    except Exception as e:
        st.error(f"Fehler bei {symbol_input}: {e}")





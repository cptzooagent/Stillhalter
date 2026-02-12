import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

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
    """Berechnet Daily und Weekly Pivot-Punkte für maximale Sicherheit."""
    try:
        tk = yf.Ticker(symbol)
        
        # 1. Daily Pivots (Vortag)
        hist_d = tk.history(period="5d") 
        if len(hist_d) < 2: return None
        last_day = hist_d.iloc[-2]
        h_d, l_d, c_d = last_day['High'], last_day['Low'], last_day['Close']
        p_d = (h_d + l_d + c_d) / 3
        s1_d = (2 * p_d) - h_d
        s2_d = p_d - (h_d - l_d)

        # 2. Weekly Pivots (Vorwoche für den "echten" Boden)
        hist_w = tk.history(period="3wk", interval="1wk")
        if len(hist_w) < 2: 
            return {"P": p_d, "S1": s1_d, "S2": s2_d, "W_S2": s2_d}
        
        last_week = hist_w.iloc[-2]
        h_w, l_w, c_w = last_week['High'], last_week['Low'], last_week['Close']
        p_w = (h_w + l_w + c_w) / 3
        s2_w = p_w - (h_w - l_w)

        return {
            "P": p_d, "S1": s1_d, "S2": s2_d, 
            "W_S2": s2_w  # Wöchentlicher starker Support
        }
    except:
        return None

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
        if hist.empty: return None, [], "", 50, True, False, 0
        price = hist['Close'].iloc[-1] 
        dates = list(tk.options)
        rsi_series = calculate_rsi(hist['Close'])
        rsi_val = rsi_series.iloc[-1]
        sma_200 = hist['Close'].mean() 
        is_uptrend = price > sma_200
        sma_20 = hist['Close'].rolling(window=20).mean()
        std_20 = hist['Close'].rolling(window=20).std()
        lower_band = (sma_20 - 2 * std_20).iloc[-1]
        is_near_lower = price <= (lower_band * 1.02)
        atr = (hist['High'] - hist['Low']).rolling(window=14).mean().iloc[-1]
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return price, dates, earn_str, rsi_val, is_uptrend, is_near_lower, atr
    except:
        return None, [], "", 50, True, False, 0

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



# --- SEKTION: PROFI-SCANNER (RSI-RISIKO & STERNE EDITION) ---
if st.button("🚀 Profi-Scan starten", key="kombi_scan_pro"):
    puffer_limit = otm_puffer_slider / 100 
    mkt_cap_limit = min_mkt_cap * 1_000_000_000
    
    with st.spinner("Lade Ticker-Liste und analysiere Setups..."):
        if test_modus:
            ticker_liste = ["APP", "AVGO", "NET", "CRWD", "MRVL", "NVDA", "CRDO", "HOOD", "SE", "ALAB", "TSLA", "PLTR", "COIN", "MSTR", "TER", "DELL", "DDOG", "MU", "LRCX", "RTX", "UBER"]
        else:
            ticker_liste = get_combined_watchlist()
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    all_results = []
    
    for i, symbol in enumerate(ticker_liste):
        progress_bar.progress((i + 1) / len(ticker_liste))
        if i % 3 == 0: 
            status_text.text(f"Analysiere {i}/{len(ticker_liste)}: {symbol}...")
        
        try:
            tk = yf.Ticker(symbol)
            info = tk.info
            curr_price = info.get('currentPrice', 0)
            m_cap = info.get('marketCap', 0)
            
            # Filter 1: Basis-Daten
            if m_cap < mkt_cap_limit: continue
            if not (min_stock_price <= curr_price <= max_stock_price): continue
            
            res = get_stock_data_full(symbol)
            if res[0] is None: continue
            price, dates, earn, rsi, uptrend, near_lower, atr = res
            
            # Filter 2: Trend
            if only_uptrend and not uptrend: continue

            # Filter 3: Laufzeit & Earnings-Schutz
            heute = datetime.now()
            max_days_allowed = 24
            if earn and "." in earn:
                try:
                    tag, monat = earn.split(".")[:2]
                    er_datum = datetime(heute.year, int(monat), int(tag))
                    if er_datum < heute: er_datum = datetime(heute.year + 1, int(monat), int(tag))
                    max_days_allowed = min(24, (er_datum - heute).days - 2)
                except: pass

            valid_dates = [d for d in dates if 11 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= max_days_allowed]
            if not valid_dates: continue
            
            # Options-Check
            target_date = valid_dates[0]
            chain = tk.option_chain(target_date).puts
            target_strike = price * (1 - puffer_limit)
            opts = chain[chain['strike'] <= target_strike].sort_values('strike', ascending=False)
            
            if not opts.empty:
                o = opts.iloc[0]
                bid_val = o['bid'] if o['bid'] > 0 else o['lastPrice']
                days = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                y_pa = (bid_val / o['strike']) * (365 / max(1, days)) * 100
                
                if y_pa >= min_yield_pa:
                    analyst_txt, analyst_col = get_analyst_conviction(info)
                    
                    # --- OPTIMIERTE STERNE-LOGIK (RISIKO-FILTER) ---
                    stars = 0
                    if "HYPER" in analyst_txt: stars = 3
                    elif "Stark" in analyst_txt: stars = 2
                    elif "Neutral" in analyst_txt: stars = 1
                    
                    # Abzug bei technischer Gefahr (z.B. HOOD RSI 24)
                    if rsi < 30: stars -= 1 
                    if rsi > 75: stars -= 0.5 # Überhitzt
                    
                    # Bonus bei stabilem Trend
                    if uptrend and stars > 0: stars += 0.5 
                    
                    stars = max(0, float(stars)) # Verhindert negative Sterne

                    all_results.append({
                        'symbol': symbol, 'price': price, 'y_pa': y_pa, 
                        'strike': o['strike'], 'puffer': ((price - o['strike']) / price) * 100,
                        'bid': bid_val, 'rsi': rsi, 'earn': earn if earn else "n.a.", 
                        'tage': days, 'status': "🛡️ Trend" if uptrend else "💎 Dip",
                        'stars_val': stars, 'stars_str': "⭐" * int(stars) if stars >= 1 else "⚠️",
                        'analyst_txt': analyst_txt, 'analyst_col': analyst_col,
                        'mkt_cap': m_cap / 1_000_000_000
                    })
        except: continue

    status_text.empty()
    progress_bar.empty()

    if not all_results:
        st.warning("Keine Treffer gefunden, die den Kriterien entsprechen.")
    else:
        # Sortierung: Qualität (Sterne) -> Rendite
        all_results = sorted(all_results, key=lambda x: (x['stars_val'], x['y_pa']), reverse=True)

        st.markdown(f"### 🎯 Top-Setups ({len(all_results)} Treffer)")
        cols = st.columns(4)
        for idx, res in enumerate(all_results):
            with cols[idx % 4]:
                # Farblogik für Karten-Design
                s_color = "#27ae60" if "🛡️" in res['status'] else "#2980b9"
                border_color = res['analyst_col'] if res['stars_val'] >= 2 else "#e0e0e0"
                rsi_col = "#e74c3c" if res['rsi'] > 70 or res['rsi'] < 30 else "#7f8c8d"
                
                with st.container(border=True):
                    # Header mit Symbol und Risiko-Check
                    st.markdown(f"**{res['symbol']}** {res['stars_str']} <span style='float:right; font-size:0.75em; color:{s_color}; font-weight:bold;'>{res['status']}</span>", unsafe_allow_html=True)
                    st.metric("Yield p.a.", f"{res['y_pa']:.1f}%")
                    
                    # Details
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

# Deine echten Depot-Werte: [Anzahl, Einstandspreis]
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
            price, dates, earn, rsi, uptrend, near_lower, atr = res
            qty, entry = data[0], data[1]
            perf_pct = ((price - entry) / entry) * 100
            
            pivots = calculate_pivots(symbol)
            s2_d = pivots['S2'] if pivots else 0
            s2_w = pivots['W_S2'] if pivots else 0
            
            # --- REPARATUR-LOGIK ---
            # 1. Short Put Signal: Wenn RSI tief ODER Kurs bei Daily S2
            put_action = "🟢 JETZT (S2/RSI)" if (rsi < 35 or price <= s2_d * 1.02) else "⏳ Warten"
            
            # Bonus-Signal: Wenn sogar der Wochen-Support erreicht ist
            if price <= s2_w * 1.01:
                put_action = "🔥 EXTREM (Weekly S2)"
            
            # 2. Covered Call Signal: Wenn Erholung eingesetzt hat
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

    st.info("💡 **Strategie:** Wenn 'Short Put' auf 🟢 steht, ist die Aktie technisch so tief, dass du durch den Verkauf eines Puts am S2-Level deinen Einstand sicher verbilligen kannst.")

                    
# --- SEKTION 3: DESIGN-UPGRADE & SICHERHEITS-AMPEL (INKL. PANIK-SCHUTZ) ---
st.markdown("### 🔍 Profi-Analyse & Trading-Cockpit")
symbol_input = st.text_input("Ticker Symbol", value="MU", help="Gib ein Ticker-Symbol ein").upper()

if symbol_input:
    try:
        with st.spinner(f"Erstelle Dashboard für {symbol_input}..."):
            tk = yf.Ticker(symbol_input)
            info = tk.info
            res = get_stock_data_full(symbol_input)
            
            if res[0] is not None:
                price, dates, earn, rsi, uptrend, near_lower, atr = res
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

                # --- HIER EINFÜGEN: PIVOT ANALYSE ---
                st.markdown("---")
                pivots = calculate_pivots(symbol_input)
                if pivots:
                    st.markdown("#### 🛡️ Technische Absicherung (Vortages-Pivots)")
                    pc1, pc2, pc3 = st.columns(3)
                    
                    pc1.markdown(f"""
                        <div style="text-align:center; padding:10px; border:1px solid #ddd; border-radius:10px; background: white;">
                            <small style="color: #7f8c8d;">Pivot Punkt (P)</small><br>
                            <b style="font-size:1.2em; color: #2c3e50;">{pivots['P']:.2f} $</b>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    pc2.markdown(f"""
                        <div style="text-align:center; padding:10px; border:2px solid #27ae60; border-radius:10px; background: #27ae6005;">
                            <small style="color: #27ae60;">Support S1</small><br>
                            <b style="font-size:1.2em; color: #27ae60;">{pivots['S1']:.2f} $</b>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    pc3.markdown(f"""
                        <div style="text-align:center; padding:10px; border:2px solid #2ecc71; border-radius:10px; background: #2ecc7105;">
                            <small style="color: #27ae60;">Support S2 (Stark)</small><br>
                            <b style="font-size:1.2em; color: #27ae60;">{pivots['S2']:.2f} $</b>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Kleiner Hinweis-Text unter den Boxen
                    st.caption(f"💡 Profi-Check: Liegt dein Strike unter S1 ({pivots['S1']:.2f} $)? Das wäre ein deutlich sichereres Setup.")
                # --- ENDE PIVOT ANALYSE ---


                # 3. ANALYSTEN BOX
                st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 10px solid {analyst_col}; margin-top: 10px;">
                        <h4 style="margin-top:0; color: #31333F;">💡 Fundamentale Analyse</h4>
                        <p style="font-size: 1.1em; font-weight: bold; color: {analyst_col};">{analyst_txt}</p>
                        <hr style="margin: 10px 0;">
                        <span style="color: #555;">📅 Nächste Earnings: <b>{earn if earn else 'n.a.'}</b></span>
                    </div>
                """, unsafe_allow_html=True)

                # 4. OPTIONEN TABELLE
                st.markdown("### 🎯 Option-Chain Auswahl")
                heute = datetime.now()
                # Flexibles Fenster: 5 bis 35 Tage
                valid_dates = [d for d in dates if 5 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 35]
                
                if valid_dates:
                    target_date = st.selectbox("📅 Wähle deinen Verfallstag", valid_dates)
                    chain = tk.option_chain(target_date).puts
                    days_to_expiry = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                    
                    chain['strike'] = chain['strike'].astype(float)
                    chain['Puffer %'] = ((price - chain['strike']) / price) * 100
                    chain['Yield p.a. %'] = (chain['bid'] / chain['strike']) * (365 / max(1, days_to_expiry)) * 100
                    
                    df_disp = chain[(chain['strike'] < price) & (chain['Puffer %'] < 25)].copy()
                    df_disp = df_disp.sort_values('strike', ascending=False)

                    def style_rows(row):
                        p = row['Puffer %']
                        if p >= 12: return ['background-color: rgba(39, 174, 96, 0.1)'] * len(row)
                        elif 8 <= p < 12: return ['background-color: rgba(241, 196, 15, 0.1)'] * len(row)
                        return ['background-color: rgba(231, 76, 60, 0.1)'] * len(row)

                    styled_df = df_disp[['strike', 'bid', 'ask', 'Puffer %', 'Yield p.a. %']].style.apply(style_rows, axis=1).format({
                        'strike': '{:.2f} $', 'bid': '{:.2f} $', 'ask': '{:.2f} $',
                        'Puffer %': '{:.1f} %', 'Yield p.a. %': '{:.1f} %'
                    })
                    
                    st.dataframe(styled_df, use_container_width=True, height=450)
                    st.caption("🟢 >12% Puffer | 🟡 8-12% Puffer | 🔴 <8% Puffer")

    except Exception as e:
        st.error(f"Fehler bei {symbol_input}: {e}")








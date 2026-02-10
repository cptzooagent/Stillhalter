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



# --- SEKTION: PROFI-SCANNER (RSI & STERNE EDITION) ---
if st.button("🚀 Profi-Scan starten", key="kombi_scan_pro"):
    puffer_limit = otm_puffer_slider / 100 
    mkt_cap_limit = min_mkt_cap * 1_000_000_000
    
    with st.spinner("Lade Ticker-Liste..."):
        if test_modus:
            # HIER SIND ALLE DEINE SYMBOLE GESICHERT:
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
            
            if m_cap < mkt_cap_limit: continue
            if not (min_stock_price <= curr_price <= max_stock_price): continue
            
            res = get_stock_data_full(symbol)
            if res[0] is None: continue
            price, dates, earn, rsi, uptrend, near_lower, atr = res
            
            if only_uptrend and not uptrend: continue

            # Laufzeit & Earnings-Schutz
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
                    
                    # Sterne Logik (Priorität)
                    stars = 0
                    if "HYPER" in analyst_txt: stars = 3
                    elif "Stark" in analyst_txt: stars = 2
                    elif "Neutral" in analyst_txt: stars = 1
                    if uptrend and stars > 0: stars += 0.5 

                    all_results.append({
                        'symbol': symbol, 'price': price, 'y_pa': y_pa, 
                        'strike': o['strike'], 'puffer': ((price - o['strike']) / price) * 100,
                        'bid': bid_val, 'rsi': rsi, 'earn': earn if earn else "n.a.", 
                        'tage': days, 'status': "🛡️ Trend" if uptrend else "💎 Dip",
                        'stars_val': stars, 'stars_str': "⭐" * int(stars),
                        'analyst_txt': analyst_txt, 'analyst_col': analyst_col,
                        'mkt_cap': m_cap / 1_000_000_000
                    })
        except: continue

    status_text.empty()
    progress_bar.empty()

    if not all_results:
        st.warning("Keine Treffer gefunden.")
    else:
        # Sortierung: Qualität (Sterne) -> Rendite
        all_results = sorted(all_results, key=lambda x: (x['stars_val'], x['y_pa']), reverse=True)

        st.markdown(f"### 🎯 Top-Setups ({len(all_results)} Treffer)")
        cols = st.columns(4)
        for idx, res in enumerate(all_results):
            with cols[idx % 4]:
                s_color = "#27ae60" if "🛡️" in res['status'] else "#2980b9"
                border_color = res['analyst_col'] if res['stars_val'] >= 2 else "#e0e0e0"
                rsi_col = "#e74c3c" if res['rsi'] > 70 else ("#27ae60" if res['rsi'] < 30 else "#7f8c8d")
                
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
                    
# --- SEKTION 4: DEPOT-MANAGER MIT BREAK-EVEN LOGIK ---
st.markdown("### 🛠️ Depot-Manager: Strategische Reparatur")

# Erstellen einer Liste von Dictionaries für dein Depot
if 'my_stocks' not in st.session_state:
    st.session_state.my_stocks = [
        {"symbol": "HIMS", "buy_price": 22.50, "shares": 100},
        {"symbol": "MU", "buy_price": 110.00, "shares": 100}
    ]

for stock in st.session_state.my_stocks:
    symbol = stock["symbol"]
    buy_price = stock["buy_price"]
    
    with st.expander(f"Position: {symbol} (Einstieg: {buy_price:.2f} $)", expanded=True):
        res = get_stock_data_full(symbol)
        if res[0] is not None:
            price, dates, earn, rsi, uptrend, near_lower, atr = res
            
            # --- BREAK-EVEN BERECHNUNG ---
            profit_loss_pct = ((price - buy_price) / buy_price) * 100
            pl_color = "#27ae60" if profit_loss_pct >= 0 else "#e74c3c"
            
            # --- SPEZIELLE REPARATUR-AMPEL ---
            if rsi < 30:
                status_txt, status_col = "🚫 WARTEN (Bodenbildung abwarten)", "#c0392b"
            elif rsi > 55:
                status_txt, status_col = "✅ CALL VERKAUFEN (Erholung nutzen)", "#27ae60"
            else:
                status_txt, status_col = "🟡 GEDULD (RSI neutral)", "#f1c40f"

            # Darstellung
            st.markdown(f"""
                <div style="background:{status_col}; color:white; padding:10px; border-radius:5px; text-align:center; font-weight:bold; margin-bottom:10px;">
                    {status_txt}
                </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Aktueller Kurs", f"{price:.2f} $")
            c2.metric("Performance", f"{profit_loss_pct:.1f} %", delta=f"{price-buy_price:.2f} $", delta_color="normal")
            c3.metric("RSI (14)", int(rsi))

            # --- CALL-STRATEGIE EMPFEHLUNG ---
            st.write("---")
            st.subheader("💡 Call-Reparatur Vorschlag")
            
            if st.checkbox(f"Optionen prüfen für {symbol}"):
                tk = yf.Ticker(symbol)
                valid_dates = [d for d in dates if 15 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= 40]
                
                if valid_dates:
                    target_date = st.selectbox(f"Laufzeit", valid_dates, key=f"d_{symbol}")
                    calls = tk.option_chain(target_date).calls
                    
                    # Nur Calls anzeigen, die ÜBER oder NAHE deinem Einstandspreis liegen
                    calls['Abstand zu Entry %'] = ((calls['strike'] - buy_price) / buy_price) * 100
                    safe_calls = calls[calls['strike'] >= buy_price * 0.95].sort_values('strike')
                    
                    st.write(f"Empfehlung: Wähle einen Strike nahe **{buy_price:.2f} $**, um bei Zuweisung keinen Verlust zu machen.")
                    
                    st.dataframe(safe_calls[['strike', 'bid', 'Abstand zu Entry %']].head(5).style.format({
                        'strike': '{:.2f} $', 'bid': '{:.2f} $', 'Abstand zu Entry %': '{:.1f} %'
                    }), use_container_width=True)
                    
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



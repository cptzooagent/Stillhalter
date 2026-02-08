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
st.sidebar.header("🛡️ Strategie-Einstellungen")
otm_puffer_slider = st.sidebar.slider("Gewünschter Puffer (%)", 3, 25, 10, help="Abstand vom Strike zum Kurs")
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", 0, 100, 15)
min_stock_price, max_stock_price = st.sidebar.slider("Aktienpreis-Spanne ($)", 0, 1000, (20, 500))

st.sidebar.markdown("---")
only_uptrend = st.sidebar.checkbox("Nur Aufwärtstrend (SMA 200)", value=False)
st.sidebar.info("Tipp: Deaktiviere den Aufwärtstrend für mehr Treffer am Wochenende.")

# --- DAS ULTIMATIVE MARKT-DASHBOARD (4 SPALTEN) ---
st.markdown("## 📊 Globales Marktwetter")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

# 1. Klasisscher Fear & Greed (Aktienmarkt) - Näherungswert über Marktdaten
# Da CNN keine API hat, nutzen wir ein stabiles Sentiment-Modell oder eine Platzhalter-Logik
with m_col1:
    # Hinweis: Da der CNN Index schwer zu scrapen ist, hier ein Indikator-Feld
    # Du kannst hier manuell den Wert prüfen oder wir nutzen RSI des SPY als Proxy
    spy_hist = yf.Ticker("SPY").history(period="20d")
    spy_rsi = calculate_rsi(spy_hist['Close']).iloc[-1]
    fng_status = "Neutral" if 40 < spy_rsi < 60 else "Greed" if spy_rsi >= 60 else "Fear"
    st.metric("Stock Fear & Greed", f"{spy_rsi:.0f}/100", f"Status: {fng_status}")

# 2. Crypto Fear & Greed (Der aus dem letzten Schritt)
try:
    import requests
    fg_data = requests.get("https://api.alternative.me/fng/").json()
    fg_crypto = int(fg_data['data'][0]['value'])
    fg_c_text = fg_data['data'][0]['value_classification']
    m_col2.metric("Crypto Fear & Greed", f"{fg_crypto}/100", fg_c_text)
except:
    m_col2.error("Crypto F&G n.a.")

# 3. VIX Abfrage
try:
    vix_val = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
    m_col3.metric("VIX (Angst-Index)", f"{vix_val:.2f}", delta_color="inverse")
except:
    m_col3.error("VIX n.a.")

# 4. Bitcoin Abfrage
try:
    btc_price = yf.Ticker("BTC-USD").history(period="1d")['Close'].iloc[-1]
    m_col4.metric("Bitcoin (Risk-On)", f"{btc_price:,.0f} $")
except:
    m_col4.error("BTC n.a.")

st.markdown("---")

# --- SEKTION 1: KOMBI-SCAN ---
if st.button("🚀 Kombi-Scan starten"):
    puffer_limit = otm_puffer_slider / 100 
    
    with st.spinner("Lade Marktliste..."):
        ticker_liste = get_combined_watchlist()
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    all_results = []
    
    for i, symbol in enumerate(ticker_liste):
        if i % 5 == 0 or i == len(ticker_liste)-1:
            progress_bar.progress((i + 1) / len(ticker_liste))
            status_text.text(f"Scanne {i+1}/{len(ticker_liste)}: {symbol}...")
        
        try:
            res = get_stock_data_full(symbol)
            if res[0] is None or not res[1]: continue
            price, dates, earn, rsi, uptrend, near_lower, atr = res
            
            if not (min_stock_price <= price <= max_stock_price): continue
            if only_uptrend and not uptrend: continue
            
            # Datums-Logik (11-20 Tage)
            available_dates = [d for d in dates if 11 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= 20]
            target_date = available_dates[-1] if available_dates else next((d for d in dates if (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days >= 11), None)
            if not target_date: continue

            tk = yf.Ticker(symbol)
            chain = tk.option_chain(target_date).puts
            max_strike = price * (1 - puffer_limit)
            secure_options = chain[chain['strike'] <= max_strike].sort_values('strike', ascending=False)
            
            if not secure_options.empty:
                best_opt = secure_options.iloc[0]
                bid = best_opt['bid'] if best_opt['bid'] > 0 else (best_opt['lastPrice'] if best_opt['lastPrice'] > 0 else 0.05)
                tage = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days
                y_pa = (bid / best_opt['strike']) * (365 / max(1, tage)) * 100
                puffer_ist = ((price - best_opt['strike']) / price) * 100
                
                if y_pa >= min_yield_pa:
                    all_results.append({
                        'symbol': symbol, 'price': price, 'y_pa': y_pa, 'strike': best_opt['strike'],
                        'puffer': puffer_ist, 'bid': bid, 'rsi': rsi, 'uptrend': uptrend,
                        'earn': earn, 'tage': tage, 'date': target_date
                    })
        except: continue

    all_results = sorted(all_results, key=lambda x: x['y_pa'], reverse=True)
    status_text.empty()
    progress_bar.empty()
    
    if not all_results:
        st.warning("Keine Treffer gefunden.")
    else:
        st.success(f"Scan beendet. {len(all_results)} Chancen sortiert nach Rendite identifiziert!")
        cols = st.columns(4)
        for idx, res in enumerate(all_results):
            with cols[idx % 4]:
                earn_warning = f" ⚠️ <span style='color:#e67e22; font-size:0.8em;'>ER: {res['earn']}</span>" if res['earn'] else ""
                rsi_color = "#e74c3c" if res['rsi'] > 70 else "#2ecc71" if res['rsi'] < 40 else "#555"
                with st.container(border=True):
                    st.markdown(f"**{res['symbol']}** {'✅' if res['uptrend'] else '📉'}{earn_warning}", unsafe_allow_html=True)
                    st.metric("Yield p.a.", f"{res['y_pa']:.1f}%")
                    st.markdown(f"""
                    <div style="font-size: 0.85em; line-height: 1.4; background-color: #f1f3f6; padding: 10px; border-radius: 8px; border-left: 5px solid #2ecc71;">
                    <b style="color: #1e7e34; font-size: 1.1em;">Prämie: {res['bid']:.2f}$</b><br>
                    <span style="color: #666;">({res['bid']*100:.0f}$ pro Kontrakt)</span><hr style="margin: 8px 0;">
                    <b>Strike:</b> {res['strike']:.1f}$ ({res['puffer']:.1f}% Puffer)<br>
                    <b>Kurs:</b> {res['price']:.2f}$ | <b>RSI:</b> <span style="color:{rsi_color}; font-weight:bold;">{res['rsi']:.0f}</span><br>
                    <b>Termin:</b> {res['tage']} Tage
                    </div>
                    """, unsafe_allow_html=True)

# --- SEKTION 2: SMART DEPOT-MANAGER ---
st.markdown("### 💼 Smart Depot-Manager")
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
    price, _, earn, rsi, uptrend, near_lower, atr = get_stock_data_full(item['Ticker'])
    if price:
        diff = (price / item['Einstand'] - 1) * 100
        perf_color = "#2ecc71" if diff >= 0 else "#e74c3c"
        with p_cols[i % 4]:
            with st.container(border=True):
                t_emoji = "📈" if uptrend else "📉"
                st.markdown(f"**{item['Ticker']}** {t_emoji} <span style='float:right; color:{perf_color}; font-weight:bold;'>{diff:+.1f}%</span>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:13px; margin:0;'>Kurs: {price:.2f}$ | RSI: {rsi:.0f}</p>", unsafe_allow_html=True)
                if diff < -15:
                    st.error("⚠️ Call-Gefahr!")
                    st.caption(f"Einstand {item['Einstand']}$ zu weit weg.")
                elif rsi > 60:
                    st.success("🟢 Call-Chance!")
                    st.caption("RSI heiß. Jetzt Calls prüfen.")
                else:
                    st.info("⏳ Warten")
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

            if mode == "put":
                filtered_df = chain[chain['strike'] <= price * 1.05].sort_values('strike', ascending=False)
            else:
                filtered_df = chain[chain['strike'] >= price * 0.95].sort_values('strike', ascending=True)
            
            st.write("---")
            for _, opt in filtered_df.head(15).iterrows():
                bid_val = opt['bid'] if not pd.isna(opt['bid']) else 0.0
                d_abs = abs(opt['delta_calc'])
                risk_emoji = "🟢" if d_abs < 0.16 else "🟡" if d_abs <= 0.30 else "🔴"
                y_pa = (bid_val / opt['strike']) * (365 / days_to_expiry) * 100
                puffer = (abs(opt['strike'] - price) / price) * 100
                bid_style = f"<span style='color:#2ecc71; font-weight:bold;'>{bid_val:.2f}$</span>"
                
                st.markdown(
                    f"{risk_emoji} **Strike: {opt['strike']:.1f}** | Bid: {bid_style} | Delta: {d_abs:.2f} | Puffer: {puffer:.1f}% | Yield: {y_pa:.1f}% p.a.",
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Fehler bei der Anzeige: {e}")



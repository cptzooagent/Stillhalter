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
    if len(data) < window + 1: return 50
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 2. DATEN-FUNKTIONEN ---
@st.cache_data(ttl=3600)
def get_combined_watchlist():
    sp500_nasdaq_mix = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ADBE", "NFLX", 
        "AMD", "INTC", "QCOM", "AMAT", "TXN", "MU", "ISRG", "LRCX", "PANW", "SNPS",
        "LLY", "V", "MA", "JPM", "WMT", "XOM", "UNH", "PG", "ORCL", "COST", 
        "ABBV", "BAC", "KO", "PEP", "CRM", "WFC", "DIS", "CAT", "AXP", "IBM",
        "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI", "MSTR"
    ]
    return list(set(sp500_nasdaq_mix))

@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        
        hist = tk.history(period="1mo")
        rsi_val = calculate_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        
        earn_date = None
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_date = cal['Earnings Date'][0]
                earn_str = earn_date.strftime('%d.%m.')
        except: pass
        
        return price, dates, earn_str, rsi_val, earn_date
    except:
        return None, [], "", 50, None

# --- UI: SEITENLEISTE ---
st.sidebar.header("🛡️ Strategie-Einstellungen")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 83)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)
sort_by_rsi = st.sidebar.checkbox("Nach RSI sortieren (Hoch -> Tief)")

# --- HAUPTBEREICH ---
st.title("🛡️ CapTrader AI Market Scanner")

# SEKTION 1: SCANNER
if st.button("🚀 Kombi-Scan starten"):
    watchlist = get_combined_watchlist()
    results = []
    prog = st.progress(0)
    status = st.empty()
    
    for i, t in enumerate(watchlist):
        status.text(f"Analysiere {t}...")
        prog.progress((i + 1) / len(watchlist))
        price, dates, earn, rsi, _ = get_stock_data_full(t)
        
        if price and dates:
            try:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                safe_opts = chain[chain['delta_val'].abs() <= max_delta].copy()
                
                if not safe_opts.empty:
                    days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                    safe_opts['y_pa'] = (safe_opts['bid'] / safe_opts['strike']) * (365 / days) * 100
                    matches = safe_opts[safe_opts['y_pa'] >= min_yield_pa]
                    
                    if not matches.empty:
                        best = matches.sort_values('y_pa', ascending=False).iloc[0]
                        results.append({
                            'ticker': t, 'yield': best['y_pa'], 'strike': best['strike'], 
                            'bid': best['bid'], 'puffer': (abs(best['strike'] - price) / price) * 100, 
                            'delta': abs(best['delta_val']), 'earn': earn, 'rsi': rsi
                        })
            except: continue

    status.text("Kombinations-Scan abgeschlossen!")
    if results:
        df_res = pd.DataFrame(results)
        sort_col = 'rsi' if sort_by_rsi else 'yield'
        opp_df = df_res.sort_values(sort_col, ascending=False).head(12)
        
        cols = st.columns(4)
        for idx, row in enumerate(opp_df.to_dict('records')):
            with cols[idx % 4]:
                st.markdown(f"### {row['ticker']}")
                st.caption(f"RSI: {row['rsi']:.0f}")
                if row['earn']: st.warning(f"⚠️ ER: {row['earn']}")
                st.metric("Rendite p.a.", f"{row['yield']:.1f}%")
                st.write(f"💰 Bid: **{row['bid']:.2f}$** | Puffer: **{row['puffer']:.1f}%**")
                st.write(f"🎯 Strike: **{row['strike']:.1f}$** (Δ {row['delta']:.2f})")
    else:
        st.warning(f"Keine Treffer.")

st.write("---") 

# SEKTION 2: DEPOT STATUS
st.subheader("💼 Smart Depot-Manager")
depot_data = [
    {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0},
    {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
    {"Ticker": "GTM", "Einstand": 17.0}, {"Ticker": "HIMS", "Einstand": 37.0},
    {"Ticker": "HOOD", "Einstand": 82.82}, {"Ticker": "JKS", "Einstand": 50.0},
    {"Ticker": "NVO", "Einstand": 97.0}, {"Ticker": "RBRK", "Einstand": 70.0},
    {"Ticker": "SE", "Einstand": 170.0}, {"Ticker": "TTD", "Einstand": 102.0}
]

p_cols = st.columns(3)
for i, item in enumerate(depot_data):
    price, _, earn, rsi, earn_dt = get_stock_data_full(item['Ticker'])
    if price:
        diff = (price / item['Einstand'] - 1) * 100
        with p_cols[i % 3]:
            with st.expander(f"{item['Ticker']} ({diff:.1f}%)", expanded=True):
                if earn_dt is not None:
                    try:
                        days_to_earn = (earn_dt.replace(tzinfo=None) - datetime.now().replace(tzinfo=None)).days
                        if 0 <= days_to_earn <= 3:
                            st.error(f"🚨 EARNINGS IN {days_to_earn} TAGEN!")
                    except: pass

                c1, c2 = st.columns(2)
                c1.metric("Kurs", f"{price:.2f}$")
                c2.metric("RSI", f"{rsi:.0f}")
                
                if diff > -5 and rsi > 65:
                    st.success("✅ Call-Verkauf prüfen")
                elif rsi < 35:
                    st.info("💎 Oversold - Hold")
                
                if earn: st.caption(f"📅 Earnings: {earn}")

st.write("---") 

# SEKTION 3: EINZEL-CHECK (FIX: DELTA WIEDER DA)
st.subheader("🔍 Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: t_in = st.text_input("Ticker", value="NVDA").upper()

if t_in:
    price, dates, earn, rsi, _ = get_stock_data_full(t_in)
    if price and dates:
        st.write(f"Kurs: **{price:.2f}$** | RSI: **{rsi:.0f}**")
        d_sel = st.selectbox("Laufzeit wählen", dates)
        tk = yf.Ticker(t_in)
        
        try:
            chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
            T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
            
            # Delta für die Filterung und Anzeige berechnen
            chain['delta_calc'] = chain.apply(lambda opt: calculate_bsm_delta(
                price, opt['strike'], T, opt['impliedVolatility'] or 0.4, option_type=mode
            ), axis=1)

            if mode == "put":
                filtered_df = chain[(chain['delta_calc'] <= -0.10) & (chain['strike'] < price)].sort_values('strike', ascending=False)
            else:
                filtered_df = chain[(chain['delta_calc'] >= 0.10) & (chain['strike'] > price)].sort_values('strike', ascending=True)
            
            for _, opt in filtered_df.iterrows():
                d_abs = abs(opt['delta_calc'])
                risk = "🟢" if d_abs < 0.16 else "🟡" if d_abs < 0.31 else "🔴"
                
                # Delta im Titel des Expanders hinzugefügt
                with st.expander(f"{risk} Strike {opt['strike']:.1f}$ | Delta: {d_abs:.2f} | Bid: {opt['bid']:.2f}$"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"💰 **Preis:** {opt['bid']:.2f}$")
                        st.write(f"📊 **OTM:** {(1-d_abs)*100:.1f}%")
                    with col_b:
                        st.write(f"🎯 **Puffer:** {(abs(opt['strike']-price)/price)*100:.1f}%")
                        st.write(f"📉 **Delta:** {d_abs:.2f}")
                        st.write(f"🌊 **IV:** {int((opt['impliedVolatility'] or 0)*100)}%")
        except Exception as e:
            st.error(f"Fehler: {e}")


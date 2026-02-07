import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

# --- SETUP & STYLING ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# Custom CSS für ein moderneres Interface
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .status-card { border-radius: 10px; padding: 20px; margin-bottom: 20px; border: 1px solid #e0e0e0; background-color: white; }
    .ticker-header { font-size: 1.5rem; font-weight: bold; color: #1e1e1e; margin-bottom: 5px; }
    .rsi-badge { padding: 4px 8px; border-radius: 5px; font-weight: bold; font-size: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)

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
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "AMD", "NFLX", 
            "LLY", "V", "MA", "COST", "CRM", "PLTR", "AFRM", "HOOD", "SQ", "MSTR"]

@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        hist = tk.history(period="1mo")
        rsi_val = calculate_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        earn_date, earn_str = None, ""
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
with st.sidebar:
    st.header("⚙️ Konfiguration")
    target_prob = st.slider("🛡️ Sicherheit (OTM %)", 70, 98, 85)
    max_delta = (100 - target_prob) / 100
    min_yield_pa = st.number_input("💵 Min. Rendite p.a. (%)", value=15)
    sort_by_rsi = st.toggle("🔄 Nach RSI sortieren", value=False)
    st.divider()
    st.caption("CapTrader AI v2.5 - Professional Edition")

# --- HAUPTBEREICH ---
st.title("🛡️ Market Intelligence Scanner")

# SEKTION 1: KOMBINE-SCANNER
if st.button("🚀 Markt-Analyse starten", use_container_width=True):
    watchlist = get_combined_watchlist()
    results = []
    prog_bar = st.progress(0)
    
    for i, t in enumerate(watchlist):
        prog_bar.progress((i + 1) / len(watchlist))
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
                        results.append({'ticker': t, 'yield': best['y_pa'], 'strike': best['strike'], 'bid': best['bid'], 
                                        'puffer': (abs(best['strike'] - price) / price) * 100, 'delta': abs(best['delta_val']), 'earn': earn, 'rsi': rsi})
            except: continue

    if results:
        df_res = pd.DataFrame(results)
        opp_df = df_res.sort_values('rsi' if sort_by_rsi else 'yield', ascending=False).head(12)
        cols = st.columns(3)
        for idx, row in enumerate(opp_df.to_dict('records')):
            with cols[idx % 3]:
                with st.container():
                    st.markdown(f"### {row['ticker']}")
                    c1, c2 = st.columns(2)
                    c1.metric("Rendite p.a.", f"{row['yield']:.1f}%")
                    c2.metric("Sicherheit", f"Δ {row['delta']:.2f}")
                    st.write(f"**Strike:** {row['strike']:.1f}$ | **Puffer:** {row['puffer']:.1f}%")
                    if row['earn']: st.caption(f"📅 Earnings: {row['earn']}")
                    st.divider()
    else:
        st.info("Keine Opportunitäten gefunden, die den Kriterien entsprechen.")

# SEKTION 2: DEPOT-MANAGER (VISUELLE KARTEN)
st.divider()
st.subheader("💼 Aktive Positionen & Portfolio-Check")
depot_data = [{"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "HOOD", "Einstand": 82.82}, {"Ticker": "NVDA", "Einstand": 110.0}]

p_cols = st.columns(3)
for i, item in enumerate(depot_data):
    price, _, earn, rsi, earn_dt = get_stock_data_full(item['Ticker'])
    if price:
        diff = (price / item['Einstand'] - 1) * 100
        with p_cols[i % 3]:
            # Farb-Indikator für RSI
            rsi_color = "#ff4b4b" if rsi > 70 else "#00c853" if rsi < 30 else "#ffa000"
            
            with st.expander(f"📊 {item['Ticker']} | {diff:+.1f}%", expanded=True):
                if earn_dt and 0 <= (earn_dt.replace(tzinfo=None) - datetime.now().replace(tzinfo=None)).days <= 3:
                    st.error(f"⚠️ Earnings in {(earn_dt.replace(tzinfo=None) - datetime.now().replace(tzinfo=None)).days} Tagen!")
                
                m1, m2 = st.columns(2)
                m1.metric("Kurs", f"{price:.2f}$")
                m2.markdown(f"**RSI (14d)**<br><span style='color:{rsi_color}; font-size:20px; font-weight:bold;'>{rsi:.0f}</span>", unsafe_allow_html=True)
                
                if diff > -5 and rsi > 65:
                    st.success("✅ Strategie: Call verkaufen")
                elif rsi < 35:
                    st.info("💎 Strategie: Hold (Oversold)")
                else:
                    st.write("Neutraler Bereich")

# SEKTION 3: EINZEL-CHECK (PREMIUM DESIGN)
st.divider()
st.subheader("🔍 Deep-Dive Einzelanalyse")
ec1, ec2, ec3 = st.columns([1, 1, 2])
with ec1: t_in = st.text_input("Ticker Symbol", value="NVDA").upper()
with ec2: mode = st.segmented_control("Optionstyp", ["put", "call"], default="put")

if t_in:
    price, dates, earn, rsi, _ = get_stock_data_full(t_in)
    if price and dates:
        st.markdown(f"**Marktpreis:** {price:.2f}$ | **RSI:** {rsi:.0f} | **Status:** {'🟢 Gesund' if rsi < 60 else '🟡 Heiß'}")
        d_sel = st.selectbox("Laufzeit wählen", dates)
        tk = yf.Ticker(t_in)
        
        try:
            chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
            T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
            chain['delta_calc'] = chain.apply(lambda opt: calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, option_type=mode), axis=1)
            
            # Filterung
            f_df = chain[(chain['delta_calc'].abs() <= 0.5) & (chain['delta_calc'].abs() >= 0.05)].sort_values('strike', ascending=(mode == "call"))
            
            for _, opt in f_df.iterrows():
                d_abs = abs(opt['delta_calc'])
                # Farblogik für Delta-Risiko
                risk_color = "🟢" if d_abs < 0.16 else "🟡" if d_abs < 0.30 else "🔴"
                
                with st.expander(f"{risk_color} Strike {opt['strike']:.1f}$ | Δ {d_abs:.2f} | Bid: {opt['bid']:.2f}$"):
                    a, b, c = st.columns(3)
                    a.write(f"**Prämie:**\n{opt['bid']*100:.0f}$")
                    b.write(f"**OTM-Prob:**\n{(1-d_abs)*100:.1f}%")
                    c.write(f"**Vola (IV):**\n{int((opt['impliedVolatility'] or 0)*100)}%")
        except Exception as e:
            st.error(f"Analyse-Fehler: {e}")

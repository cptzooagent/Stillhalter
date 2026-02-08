import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import engine as eng # Wir importieren unsere engine.py

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        hist = tk.history(period="1mo")
        rsi_val = eng.calculate_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        earn_dt = eng.get_clean_earnings(tk)
        earn_str = earn_dt.strftime('%d.%m.') if earn_dt else ""
        return price, dates, earn_str, rsi_val, earn_dt
    except:
        return None, [], "", 50, None

# --- SIDEBAR ---
st.sidebar.header("🛡️ Strategie-Einstellungen")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 83)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)
sort_by_rsi = st.sidebar.checkbox("Nach RSI sortieren")

st.title("🛡️ CapTrader AI Market Scanner")

# --- SEKTION 1: KOMBI-SCAN ---
if st.button("🚀 Kombi-Scan starten"):
    watchlist = eng.get_watchlist()
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
                
                chain['delta_val'] = chain.apply(lambda r: eng.calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
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

    status.text("Scan abgeschlossen!")
    if results:
        df_res = pd.DataFrame(results).sort_values('rsi' if sort_by_rsi else 'yield', ascending=not sort_by_rsi)
        cols = st.columns(4)
        for idx, row in enumerate(df_res.head(12).to_dict('records')):
            with cols[idx % 4]:
                st.markdown(f"### {row['ticker']}")
                st.metric("Rendite p.a.", f"{row['yield']:.1f}%")
                st.write(f"🎯 Strike: **{row['strike']:.1f}$** (Δ {row['delta']:.2f})")
                if row['earn']: st.warning(f"⚠️ ER: {row['earn']}")

st.divider()

# --- SEKTION 2: DEPOT-MANAGER ---
st.subheader("💼 Smart Depot-Manager")
depot_data = [{"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "HOOD", "Einstand": 82.82}, {"Ticker": "PLTR", "Einstand": 25.0}] # Gekürzt zur Demo

p_cols = st.columns(3)
for i, item in enumerate(depot_data):
    price, _, earn, rsi, earn_dt = get_stock_data_full(item['Ticker'])
    if price:
        diff = (price / item['Einstand'] - 1) * 100
        with p_cols[i % 3]:
            with st.expander(f"{item['Ticker']} ({diff:.1f}%)", expanded=True):
                st.metric("Kurs", f"{price:.2f}$", f"{diff:.1f}%")
                if rsi > 65: st.success("✅ Call-Verkauf prüfen")
                if rsi < 35: st.info("💎 Oversold - Hold")

st.divider()

# --- SEKTION 3: EINZEL-CHECK ---
st.subheader("🔍 Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: t_in = st.text_input("Ticker", value="NVDA").upper()

if t_in:
    price, dates, earn, rsi, _ = get_stock_data_full(t_in)
    if price and dates:
        d_sel = st.selectbox("Laufzeit wählen", dates)
        tk = yf.Ticker(t_in)
        chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
        T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
        
        chain['delta_calc'] = chain.apply(lambda opt: eng.calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, option_type=mode), axis=1)
        
        for _, opt in chain.iloc[10:20].iterrows(): # Zeigt einen Ausschnitt
            st.write(f"Strike: {opt['strike']}$ | Bid: {opt['bid']}$ | Delta: {abs(opt['delta_calc']):.2f}")

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP & MATH ---
st.set_page_config(page_title="CapTrader AI Guard & Repair", layout="wide")

def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    T = max(T, 0.0001) 
    sigma = max(sigma, 0.0001)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

@st.cache_data(ttl=900)
def get_stock_basics(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        earn_date = None
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_date = cal['Earnings Date'][0]
        except: pass
        return price, dates, earn_date
    except: return None, [], None

# --- KI REPAIR LOGIK ---
def suggest_repair_call(ticker, current_price, entry_price, dates):
    """Sucht einen Call, der Prämie bringt, ohne die Aktie zu weit unter Einstand zu verlieren."""
    if not dates: return None
    try:
        tk = yf.Ticker(ticker)
        # Wir suchen eine Laufzeit in ca. 30-45 Tagen
        target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 35))
        chain = tk.option_chain(target_date).calls
        
        # Filter: Strike sollte idealerweise am Einstand liegen, aber mind. über aktuellem Kurs
        # Wir suchen einen Call mit Delta ~ 0.30 für gute Prämie
        T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
        chain['delta'] = chain.apply(lambda r: calculate_bsm_delta(current_price, r['strike'], T, r['impliedVolatility'] or 0.4, 'call'), axis=1)
        
        # Suche Call nahe Delta 0.3 (guter Kompromiss aus Prämie/Risiko)
        best_repair = chain[chain['delta'] <= 0.35].sort_values('delta', ascending=False).iloc[0]
        return {
            'strike': best_repair['strike'],
            'bid': best_repair['bid'],
            'date': target_date,
            'desc': f"Call @{best_repair['strike']}$ bringt {best_repair['bid']*100:.0f}$ Cash"
        }
    except: return None

# --- UI SIDEBAR ---
st.sidebar.header("🛡️ Scanner-Einstellungen")
target_prob = st.sidebar.slider("Sicherheit OTM (%)", 70, 98, 85)
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)
min_stock_p = st.sidebar.number_input("Mindestkurs ($)", value=40)
max_stock_p = st.sidebar.number_input("Maximalkurs ($)", value=600)

# --- HAUPTBEREICH ---
st.title("🤖 CapTrader AI Portfolio Guard")

# SEKTION 1: KI DEPOT-ANALYSE & REPAIR
st.subheader("💼 Aktive Depot-Überwachung")
depot_data = [
    {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0},
    {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
    {"Ticker": "GTM", "Einstand": 17.0}, {"Ticker": "HIMS", "Einstand": 37.0},
    {"Ticker": "HOOD", "Einstand": 120.0}, {"Ticker": "JKS", "Einstand": 50.0},
    {"Ticker": "NVO", "Einstand": 97.0}, {"Ticker": "RBRK", "Einstand": 70.0},
    {"Ticker": "SE", "Einstand": 170.0}, {"Ticker": "TTD", "Einstand": 102.0}
]

repair_needed = []
cols = st.columns(3)

for i, item in enumerate(depot_data):
    price, dates, earn = get_stock_basics(item['Ticker'])
    if price:
        perf = (price / item['Einstand'] - 1) * 100
        with cols[i % 3]:
            with st.container(border=True):
                st.write(f"**{item['Ticker']}** (Einstand: {item['Einstand']}$)")
                st.metric("Kurs", f"{price:.2f}$", f"{perf:.1f}%")
                
                if perf < -20:
                    st.error("🚨 Repair-Status: Aktiv")
                    repair = suggest_repair_call(item['Ticker'], price, item['Einstand'], dates)
                    if repair:
                        st.info(f"💡 **KI-Tipp (Covered Call):**\nVerkaufe den {repair['strike']}$ Call ({repair['date']}).\nEinnahme: **{repair['bid']*100:.0f}$**")
                        repair_needed.append(f"{item['Ticker']}: {repair['desc']}")
                
                if earn:
                    days = (earn.date() - datetime.now().date()).days
                    if 0 <= days <= 7:
                        st.warning(f"📅 Earnings in {days} Tagen!")

st.divider()

# SEKTION 2: SCANNER FÜR NEUE TRADES
st.subheader("🔍 Markt-Scanner (Neue Gelegenheiten)")
if st.button("🚀 Markt nach Chancen scannen"):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "HOOD", "PYPL", "SQ"]
    results = []
    prog = st.progress(0)
    for i, t in enumerate(watchlist):
        prog.progress((i+1)/len(watchlist))
        price, dates, earn = get_stock_basics(t)
        if price and min_stock_p <= price <= max_stock_p and dates:
            try:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = max((datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days, 1) / 365
                chain['delta'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4, 'put'), axis=1)
                matches = chain[(chain['delta'].abs() <= (100-target_prob)/100) & (chain['bid'] > 0)].copy()
                if not matches.empty:
                    best = matches.sort_values('bid', ascending=False).iloc[0]
                    y_pa = (best['bid'] / best['strike']) * (365 / (T*365)) * 100
                    if y_pa >= min_yield_pa:
                        results.append({'Ticker': t, 'Rendite p.a.': f"{y_pa:.1f}%", 'Strike': best['strike'], 'Prämie': best['bid']*100, 'Puffer': f"{(abs(best['strike']-price)/price)*100:.1f}%"})
            except: continue
    
    if results:
        st.table(pd.DataFrame(results))
    else:
        st.warning("Keine neuen Chancen gefunden.")

import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="Pro Stillhalter Dashboard", layout="wide")

# API Keys sicher aus den Secrets laden
MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- 1. DATEN-FUNKTIONEN MIT CACHING (1 STUNDE) ---
# Diese Funktionen speichern die Ergebnisse lokal, um dein 10.000er Limit zu schützen

@st.cache_data(ttl=3600)
def get_cached_expirations(symbol):
    """Holt alle verfügbaren Ablaufdaten für Optionen"""
    try:
        url = f"https://api.marketdata.app/v1/options/expirations/{symbol}/?token={MD_KEY}"
        r = requests.get(url).json()
        if r.get('s') == 'ok':
            return sorted(r.get('expirations', []))
    except Exception as e:
        st.error(f"Fehler bei Expirations für {symbol}: {e}")
    return []

@st.cache_data(ttl=3600)
def get_cached_chain(symbol, date_str, side):
    """Holt die gesamte Optionskette für ein Datum"""
    try:
        params = {"token": MD_KEY, "side": side, "expiration": date_str}
        url = f"https://api.marketdata.app/v1/options/chain/{symbol}/"
        r = requests.get(url, params=params).json()
        if r.get('s') == 'ok':
            # Wir nutzen 'bid', weil das der echte Preis ist, den du bei CapTrader kriegst
            return pd.DataFrame({
                'strike': r['strike'], 
                'bid': r.get('bid', r['mid']), # Falls bid fehlt, nimm mid
                'delta': r.get('delta', [0.0]*len(r['strike'])), 
                'iv': r.get('iv', [0.0]*len(r['strike']))
            })
    except Exception as e:
        st.error(f"Fehler bei Chain für {symbol}: {e}")
    return None

def get_live_price(symbol):
    """Holt den aktuellen Aktienkurs via Finnhub"""
    try:
        url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}'
        r = requests.get(url).json()
        if 'c' in r and r['c'] != 0:
            return float(r['c'])
    except:
        pass
    return None

# --- 2. HAUPT-DASHBOARD UI ---
st.title("🛡️ CapTrader Pro Stillhalter Dashboard")
st.markdown(f"**Status:** Daten-Sparmodus aktiv (Cache: 1 Stunde) | Marktzeit: {datetime.now().strftime('%H:%M:%S')}")

# --- 3. TOP 10 OPPORTUNITIES ---
st.subheader("💎 Top 10 High-IV Put Gelegenheiten (Delta ~0.15)")
watchlist = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI"]

if st.button("🚀 Markt-Scan jetzt starten"):
    opps = []
    progress_bar = st.progress(0)
    
    for i, t in enumerate(watchlist):
        progress_bar.progress((i + 1) / len(watchlist))
        dates = get_cached_expirations(t)
        
        if dates:
            # Suche Datum zwischen 20 und 50 Tagen Laufzeit
            target_date = next((d for d in dates if 20 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.today()).days <= 50), dates[0])
            df = get_cached_chain(t, target_date, "put")
            
            if df is not None and not df.empty:
                # Finde den Strike, der am nächsten an Delta 0.15 liegt
                df['diff'] = (df['delta'].abs() - 0.15).abs()
                best = df.sort_values('diff').iloc[0].to_dict()
                
                days = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.today()).days
                # Rendite p.a. Berechnung basierend auf Bid-Preis
                yield_pa = (best['bid'] / best['strike']) * (365 / max(1, days)) * 100
                
                best.update({'ticker': t, 'days': days, 'yield': yield_pa, 'date': target_date})
                opps.append(best)
    
    if opps:
        opp_df = pd.DataFrame(opps).sort_values('yield', ascending=False).head(10)
        cols = st.columns(5)
        for idx, (_, row) in enumerate(opp_df.iterrows()):
            with cols[idx % 5]:
                with st.container(border=True):
                    st.write(f"### {row['ticker']}")
                    st.metric("Yield p.a.", f"{row['yield']:.1f}%")
                    st.write(f"**Strike: {row['strike']:.1f}$**")
                    st.write(f"Bid: {row['bid']:.2f}$")
                    st.caption(f"Laufzeit: {row['date']} ({row['days']} T.)")
    else:
        st.warning("Keine Daten gefunden. Evtl. API Limit erreicht?")

st.divider()

# --- 4. PORTFOLIO REPAIR LISTE ---
st.subheader("💼 Aktuelles Portfolio & Repair-Ampel")

if 'portfolio' not in st.session_state:
    # Alle deine Werte aus den Screenshots
    st.session_state.portfolio = pd.DataFrame([
        {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0}, 
        {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
        {"Ticker": "GTM", "Einstand": 17.0}, {"Ticker": "HIMS", "Einstand": 37.0},
        {"Ticker": "HOOD", "Einstand": 120.0}, {"Ticker": "JKS", "Einstand": 50.0},
        {"Ticker": "NVO", "Einstand": 97.0}, {"Ticker": "RBRK", "Einstand": 70.0},
        {"Ticker": "SE", "Einstand": 170.0}, {"Ticker": "TTD", "Einstand": 102.0}
    ])

# Tabellarische Ansicht zum Editieren
st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic", use_container_width=True)

# Visualisierung der Ampel
cols_port = st.columns(2)
for i, (_, row) in enumerate(st.session_state.portfolio.iterrows()):
    curr = get_live_price(row['Ticker'])
    with (cols_port[0] if i % 2 == 0 else cols_port[1]):
        if curr:
            diff = (curr / row['Einstand'] - 1) * 100
            # Farblogik nach deinen Vorgaben
            if diff >= 0:
                color, label, note = "🟢", "PROFIT", "Status OK"
            elif diff > -20:
                color, label, note = "🟡", "REPAIR", "Verkaufe Call Delta 0.10"
            else:
                color, label, note = "🔵", "DEEP REPAIR", "Verkaufe Call Delta 0.05"
            
            st.info(f"{color} **{row['Ticker']}**: Kurs {curr:.2f}$ ({diff:.1f}%) | Einstand: {row['Einstand']:.1f}$ \n\n **Strategie:** `{label}` ({note})")

st.divider()

# --- 5. INDIVIDUELLER OPTIONS FINDER ---
st.subheader("🔍 Detail-Analyse (Einzelwerte)")
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    find_ticker = st.text_input("Ticker-Symbol", value="HOOD").upper()
with c2:
    option_mode = st.selectbox("Optionstyp", ["put", "call"])

if find_ticker:
    live_p = get_live_price(find_ticker)
    if live_p:
        st.write(f"Aktueller Kurs von {find_ticker}: **{live_p:.2f}$**")
        all_dates = get_cached_expirations(find_ticker)
        
        if all_dates:
            chosen_date = st.selectbox("Ablaufdatum wählen", all_dates)
            chain_df = get_cached_chain(find_ticker, chosen_date, option_mode)
            
            if chain_df is not None:
                # Nur OTM Optionen zeigen (Puts unter Kurs, Calls über Kurs)
                if option_mode == "put":
                    chain_df = chain_df[chain_df['strike'] < live_p].sort_values('strike', ascending=False)
                else:
                    chain_df = chain_df[chain_df['strike'] > live_p].sort_values('strike', ascending=True)
                
                # Top 8 anzeigen
                for _, opt in chain_df.head(8).iterrows():
                    d_val = abs(opt['delta'])
                    # Risiko-Farbe
                    risk_c = "🟢" if d_val < 0.16 else "🟡" if d_val < 0.31 else "🔴"
                    with st.expander(f"{risk_c} Strike {opt['strike']:.1f}$ | Bid: {opt['bid']:.2f}$ | Delta: {d_val:.2f}"):
                        st.write(f"Wahrscheinlichkeit für Gewinn: **{(1-d_val)*100:.1f}%**")
                        st.write(f"Prämie pro Kontrakt: **{opt['bid']*100:.0f}$**")

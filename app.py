import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP & DESIGN ---
st.set_page_config(page_title="Stillhalter Pro Scanner", layout="wide")

# Keys aus den Streamlit Secrets laden
MD_KEY = st.secrets["MARKETDATA_KEY"]
FINNHUB_KEY = st.secrets["FINNHUB_KEY"]

# --- FUNKTIONEN ---

def get_live_price(symbol):
    """Holt den aktuellen Kurs stabil über Finnhub"""
    try:
        url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}'
        r = requests.get(url).json()
        if r.get('c'):
            return float(r['c'])
    except:
        return None
    return None

def get_marketdata_options(symbol):
    """Holt die Optionskette von MarketData.app"""
    url = f"https://api.marketdata.app/v1/options/chain/{symbol}/"
    # Wir laden Puts, die 'out of the money' sind
    params = {"token": MD_KEY, "side": "put", "range": "otm"}
    try:
        response = requests.get(url, params=params).json()
        if response.get('s') == 'ok':
            df = pd.DataFrame({
                'strike': response['strike'],
                'mid': response['mid'],
                'expiration': response['expiration'],
                'delta': response.get('delta', [0] * len(response['strike'])),
                'iv': response.get('iv', [0] * len(response['strike']))
            })
            return df
    except:
        return None
    return None

def format_expiry_label(val):
    """Wandelt Timestamps oder Text in lesbare Daten um (für das Menü)"""
    try:
        if isinstance(val, (int, float)):
            return datetime.fromtimestamp(val).strftime('%d.%m.%Y')
        return val
    except:
        return str(val)

# --- APP INTERFACE ---

st.title("🛡️ Pro Stillhalter Scanner")

# Favoriten für schnelle Bedienung am Handy
watchlist = ["AAPL", "TSLA", "NVDA", "MSFT", "AMD", "META"]
selected_fav = st.pills("Schnellauswahl", watchlist)
user_input = st.text_input("Ticker manuell", "")

# Ticker Logik: Favorit gewinnt vor manuellem Input
ticker = (selected_fav if selected_fav else user_input).strip().upper()

if ticker:
    price = get_live_price(ticker)
    
    if price:
        st.metric(f"Aktueller Kurs {ticker}", f"{price:.2f} $")
        
        with st.spinner('Lade Optionsdaten...'):
            chain = get_marketdata_options(ticker)
        
        if chain is not None and not chain.empty:
            # Ablaufdaten für das Dropdown vorbereiten
            expirations = sorted(chain['expiration'].unique())
            selected_expiry = st.selectbox(
                "Ablaufdatum wählen", 
                expirations, 
                format_func=format_expiry_label
            )
            
            # Daten für gewähltes Datum filtern
            df_expiry = chain[chain['expiration'] == selected_expiry].copy()
            
            st.subheader(f"Puts für {format_expiry_label(selected_expiry)}")

            # Ergebnisse anzeigen
            for _, row in df_expiry.sort_values('strike', ascending=False).head(10).iterrows():
                # --- DTE BERECHNUNG (Fix für deinen Fehler) ---
                try:
                    if isinstance(row['expiration'], (int, float)):
                        exp_dt = datetime.fromtimestamp(row['expiration'])
                    else:
                        exp_dt = datetime.strptime(str(row['expiration']), '%Y-%m-%d')
                    dte = (exp_dt - datetime.now()).days
                except:
                    dte = 30 # Fallback
                
                # Rendite & Puffer
                ann_return = (row['mid'] / row['strike']) * (365 / max(1, dte)) * 100
                puffer = ((price / row['strike']) - 1) * 100
                delta_val = abs(row['delta'])
                
                # --- DELTA AMPEL ---
                if delta_val < 0.16:
                    label = "🟢 Sicher (Konservativ)"
                elif delta_val < 0.25:
                    label = "🟡 Moderat (Standard)"
                else:
                    label = "🔴 Aggressiv (Riskant)"

                with st.expander(f"Strike {row['strike']}$ | Delta: {delta_val:.2f} | {ann_return:.1f}% p.a."):
                    st.write(f"**Risiko-Check:** {label}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Prämie (Mid)", f"{row['mid']}$")
                    c2.metric("Puffer", f"{puffer:.1f}%")
                    c3.metric("Laufzeit", f"{dte} Tage")
                    
                    st.progress(min(max(delta_val * 4, 0.0), 1.0), help="Risiko-Visualisierung basierend auf Delta")
        else:
            st.warning("Keine OTM-Puts gefunden. Eventuell sind die API-Credits für heute verbraucht.")
    else:
        st.error(f"Konnte keinen Kurs für '{ticker}' finden. Bitte Ticker prüfen.")

# ANALYSE LINKS
st.divider()
st.caption("Externe Analyse-Tools")
col1, col2, col3 = st.columns(3)
col1.link_button("Onvista", f"https://www.onvista.de/suche/?searchTerm={ticker}")
col2.link_button("Stock3", f"https://stock3.com/aktien/{ticker}-aktie")
col3.link_button("TradingView", f"https://www.tradingview.com/symbols/{ticker}/")

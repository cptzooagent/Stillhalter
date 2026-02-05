import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- KONFIGURATION ---
st.set_page_config(page_title="Stillhalter Pro Scanner", layout="wide")

# Keys aus den Streamlit Secrets
MD_KEY = st.secrets["MARKETDATA_KEY"]
FINNHUB_KEY = st.secrets["FINNHUB_KEY"]

# --- FUNKTIONEN ---

def get_live_price(symbol):
    """Holt den aktuellen Kurs über Finnhub"""
    try:
        url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}'
        r = requests.get(url).json()
        return float(r['c']) if r.get('c') else None
    except:
        return None

def get_marketdata_options(symbol):
    """Holt die Optionskette von MarketData.app"""
    url = f"https://api.marketdata.app/v1/options/chain/{symbol}/"
    params = {"token": MD_KEY, "side": "put", "range": "otm"}
    try:
        response = requests.get(url, params=params).json()
        if response.get('s') == 'ok':
            return pd.DataFrame({
                'strike': response['strike'],
                'mid': response['mid'],
                'expiration': response['expiration'],
                'delta': response.get('delta', [0] * len(response['strike'])),
                'iv': response.get('iv', [0] * len(response['strike']))
            })
    except:
        return None
    return None

def format_expiry_label(val):
    """FIX FÜR BILD 9: Wandelt Unix-Zahlen in lesbare Daten um"""
    try:
        # Wenn es eine Zahl ist (Unix Timestamp)
        if isinstance(val, (int, float)) or (isinstance(val, str) and val.isdigit()):
            return datetime.fromtimestamp(int(val)).strftime('%d.%m.%Y')
        return str(val)
    except:
        return str(val)

# --- USER INTERFACE ---
st.title("🛡️ Pro Stillhalter Scanner")

# Favoriten-Leiste
watchlist = ["AAPL", "TSLA", "NVDA", "MSFT", "AMD", "META"]
sel_fav = st.pills("Schnellauswahl", watchlist)
user_input = st.text_input("Ticker manuell", "")
ticker = (sel_fav if sel_fav else user_input).strip().upper()

if ticker:
    price = get_live_price(ticker)
    if price:
        st.metric(f"Aktueller Kurs {ticker}", f"{price:.2f} $")
        
        with st.spinner('Lade Optionskette...'):
            chain = get_marketdata_options(ticker)
        
        if chain is not None and not chain.empty:
            # Dropdown mit Datums-Fix
            expirations = sorted(chain['expiration'].unique())
            selected_expiry = st.selectbox(
                "Ablaufdatum wählen", 
                expirations, 
                format_func=format_expiry_label
            )
            
            # Filterung
            df_expiry = chain[chain['expiration'] == selected_expiry].copy()
            st.subheader(f"Puts für {format_expiry_label(selected_expiry)}")

            for _, row in df_expiry.sort_values('strike', ascending=False).head(10).iterrows():
                # --- DTE BERECHNUNG FIX ---
                try:
                    raw_exp = row['expiration']
                    if isinstance(raw_exp, (int, float)) or (isinstance(raw_exp, str) and raw_exp.isdigit()):
                        exp_dt = datetime.fromtimestamp(int(raw_exp))
                    else:
                        exp_dt = datetime.strptime(str(raw_exp), '%Y-%m-%d')
                    dte = (exp_dt - datetime.now()).days
                except:
                    dte = 30 # Notlösung
                
                # Werte berechnen
                ann_return = (row['mid'] / row['strike']) * (365 / max(1, dte)) * 100
                delta_val = abs(row['delta'])
                puffer = ((price / row['strike']) - 1) * 100
                
                # Risiko-Ampel
                color = "🟢" if delta_val < 0.16 else "🟡" if delta_val < 0.25 else "🔴"

                with st.expander(f"{color} Strike {row['strike']}$ | Delta: {delta_val:.2f} | {ann_return:.1f}% p.a."):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Prämie", f"{row['mid']}$")
                    c2.metric("Puffer", f"{puffer:.1f}%")
                    c3.metric("Laufzeit", f"{dte} Tage")
        else:
            st.warning("Keine Daten gefunden. API-Limit erreicht?")
    else:
        st.error(f"Konnte Kurs für '{ticker}' nicht laden.")

st.divider()
st.link_button("Chart öffnen (TradingView)", f"https://www.tradingview.com/symbols/{ticker}/")

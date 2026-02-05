import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP & DESIGN ---
st.set_page_config(page_title="Stillhalter Pro Scanner", layout="wide")

# Keys aus den Streamlit Secrets
MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- FUNKTIONEN ---

def get_live_price(symbol):
    """Holt den aktuellen Kurs über Finnhub"""
    try:
        url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}'
        r = requests.get(url).json()
        if r.get('c'):
            return float(r['c'])
    except:
        return None
    return None

def get_marketdata_options(symbol):
    """Holt die KOMPLETTE Optionskette von MarketData.app"""
    # WICHTIG: Wir entfernen 'range=otm', um wirklich alle verfügbaren Daten zu erhalten
    url = f"https://api.marketdata.app/v1/options/chain/{symbol}/"
    params = {
        "token": MD_KEY, 
        "side": "put"
    }
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
            
            # Umwandlung der Unix-Zahlen in Text-Datum (YYYY-MM-DD)
            def fix_date(x):
                try:
                    if isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit()):
                        return datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d')
                    return str(x)
                except:
                    return str(x)
            
            df['expiration'] = df['expiration'].apply(fix_date)
            return df
    except:
        return None
    return None

def format_expiry_with_dte(date_str):
    """Formatierungsfunktion für das Dropdown (Datum + DTE)"""
    try:
        exp_dt = datetime.strptime(date_str, '%Y-%m-%d')
        days_to_expiry = (exp_dt - datetime.now()).days
        return f"{exp_dt.strftime('%d.%m.%Y')} ({max(0, days_to_expiry)} Tage)"
    except:
        return date_str

# --- USER INTERFACE ---
st.title("🛡️ Pro Stillhalter Scanner")

# Favoriten-Leiste
watchlist = ["AAPL", "TSLA", "NVDA", "MSFT", "AMD", "META", "GOOGL", "AMZN"]
sel_fav = st.pills("Schnellauswahl", watchlist)
user_input = st.text_input("Ticker manuell (z.B. MSTR)", "")

ticker = (sel_fav if sel_fav else user_input).strip().upper()

if ticker:
    price = get_live_price(ticker)
    
    if price:
        st.metric(f"Aktueller Kurs {ticker}", f"{price:.2f} $")
        
        with st.spinner(f'Lade alle verfügbaren Ketten für {ticker}...'):
            chain = get_marketdata_options(ticker)
        
        if chain is not None and not chain.empty:
            # Wir filtern hier manuell auf OTM (Strikes < Aktueller Kurs)
            otm_chain = chain[chain['strike'] < price].copy()
            
            # Alle Verfallstage extrahieren und sortieren
            expirations = sorted(otm_chain['expiration'].unique())
            
            # Falls die Liste sehr lang ist, hilft die Anzeige der Tage im Dropdown enorm
            selected_expiry = st.selectbox(
                "Wähle Laufzeit (Datum & Resttage)", 
                expirations, 
                format_func=format_expiry_with_dte
            )
            
            df_expiry = otm_chain[otm_chain['expiration'] == selected_expiry].copy()
            
            st.subheader(f"Puts für {format_expiry_with_dte(selected_expiry)}")

            # Anzeige der Strikes (Top 15, sortiert nach Strike-Höhe)
            for _, row in df_expiry.sort_values('strike', ascending=False).head(15).iterrows():
                try:
                    exp_dt = datetime.strptime(row['expiration'], '%Y-%m-%d')
                    dte = max(1, (exp_dt - datetime.now()).days)
                except:
                    dte = 30
                
                # Renditeberechnung
                ann_return = (row['mid'] / row['strike']) * (365 / dte) * 100
                delta_val = abs(row['delta'])
                puffer = ((price / row['strike']) - 1) * 100
                
                # Risiko-Ampel
                if delta_val < 0.16:
                    color = "🟢"
                elif delta_val < 0.25:
                    color = "🟡"
                else:
                    color = "🔴"

                with st.expander(f"{color} Strike {row['strike']:.1f}$ | Delta: {delta_val:.2f} | {ann_return:.1f}% p.a."):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Prämie (Mid)", f"{row['mid']:.2f}$")
                    c2.metric("Puffer", f"{puffer:.1f}%")
                    c3.metric("Laufzeit", f"{dte} Tage")
                    
                    st.write(f"Voraussichtliche Rendite (annualisiert): **{ann_return:.1f}%**")
        else:
            st.warning("Keine Daten gefunden. Evtl. Ticker prüfen oder API-Limit erreicht.")
    else:
        st.error(f"Konnte Kurs für '{ticker}' nicht laden.")

st.divider()
st.caption("Externe Analyse:")
c1, c2, c3 = st.columns(3)
c1.link_button("OptionStrat", f"https://optionstrat.com/visualizer/cash-secured-put/{ticker}")
c2.link_button("TradingView", f"https://www.tradingview.com/symbols/{ticker}/")
c3.link_button("Stock3", f"https://stock3.com/aktien/{ticker}-aktie")

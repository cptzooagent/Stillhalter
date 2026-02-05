import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="Stillhalter Pro Scanner", layout="wide")

MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

def get_live_price(symbol):
    try:
        url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}'
        r = requests.get(url).json()
        return float(r['c']) if r.get('c') else None
    except: return None

def get_marketdata_options(symbol):
    url = f"https://api.marketdata.app/v1/options/chain/{symbol}/"
    params = {"token": MD_KEY, "side": "put", "range": "otm"}
    try:
        response = requests.get(url, params=params).json()
        if response.get('s') == 'ok':
            df = pd.DataFrame({
                'strike': response['strike'],
                'mid': response['mid'],
                'expiration': response['expiration'],
                'delta': response.get('delta', [0] * len(response['strike']))
            })
            
            # DER FIX: Alle Zeitstempel sofort in echtes Datum umwandeln
            def fix_date(x):
                try:
                    if isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit()):
                        return datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d')
                    return str(x)
                except: return str(x)
            
            df['expiration'] = df['expiration'].apply(fix_date)
            return df
    except: return None
    return None

# --- UI ---
st.title("🛡️ Pro Stillhalter Scanner")

watchlist = ["AAPL", "TSLA", "NVDA", "MSFT", "AMD", "META"]
sel_fav = st.pills("Schnellauswahl", watchlist)
ticker = (sel_fav if sel_fav else st.text_input("Ticker", "AAPL")).strip().upper()

if ticker:
    price = get_live_price(ticker)
    if price:
        st.metric(f"Kurs {ticker}", f"{price:.2f} $")
        chain = get_marketdata_options(ticker)
        
        if chain is not None and not chain.empty:
            # Jetzt sind alle Daten in 'chain' garantiert Text-Daten (YYYY-MM-DD)
            expirations = sorted(chain['expiration'].unique())
            
            # Schönere Anzeige im Dropdown
            selected_expiry = st.selectbox(
                "Ablaufdatum wählen", 
                expirations,
                format_func=lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%d.%m.%Y')
            )
            
            df_expiry = chain[chain['expiration'] == selected_expiry].copy()
            
            for _, row in df_expiry.sort_values('strike', ascending=False).head(12).iterrows():
                # DTE Berechnung ist jetzt sicher
                exp_dt = datetime.strptime(row['expiration'], '%Y-%m-%d')
                dte = max(1, (exp_dt - datetime.now()).days)
                
                ann_return = (row['mid'] / row['strike']) * (365 / dte) * 100
                delta_val = abs(row['delta'])
                
                color = "🔴" if delta_val > 0.25 else "🟡" if delta_val > 0.15 else "🟢"
                
                with st.expander(f"{color} Strike {row['strike']}$ | Delta: {delta_val:.2f} | {ann_return:.1f}% p.a."):
                    c1, c2 = st.columns(2)
                    c1.metric("Prämie", f"{row['mid']}$")
                    c2.metric("Laufzeit", f"{dte} Tage")
        else:
            st.warning("Keine Daten. API-Limit erreicht?")
    else:
        st.error(f"Ticker {ticker} nicht gefunden.")

import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime

st.set_page_config(page_title="Stillhalter Pro", layout="centered")

# --- MULTI-QUELLE DATEN LOGIK ---
def get_price(symbol):
    # Versuch 1: Yahoo (Live)
    try:
        t = yf.Ticker(symbol)
        return t.fast_info['last_price'], "Yahoo"
    except:
        pass
    
    # Versuch 2: Finnhub (API)
    try:
        key = st.secrets["FINNHUB_KEY"]
        url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={key}'
        r = requests.get(url).json()
        if r['c'] > 0: return r['c'], "Finnhub"
    except:
        pass

    # Versuch 3: Alpha Vantage (API)
    try:
        key = st.secrets["ALPHA_VANTAGE_KEY"]
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={key}'
        r = requests.get(url).json()
        return float(r['Global Quote']['05. price']), "Alpha Vantage"
    except:
        return None, "Error"

st.title("🛡️ Stillhalter Scanner")

ticker = st.text_input("Ticker Symbol", "AAPL").upper()

if ticker:
    price, source = get_price(ticker)
    
    if price:
        st.metric("Kurs", f"{price:.2f} $", help=f"Quelle: {source}")
        
        # Optionsketten (Diese laden wir weiterhin über Yahoo, da sie dort am besten sind)
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            
            if expirations:
                expiry = st.selectbox("Ablaufdatum", expirations)
                dte = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
                
                chain = stock.option_chain(expiry).puts
                # Zeige nur Puts, die "Out of the Money" sind
                otm_puts = chain[chain['strike'] < price].sort_values('strike', ascending=False).head(5)
                
                st.subheader("Beste Stillhalter-Chancen")
                for _, row in otm_puts.iterrows():
                    # Rendite p.a. berechnen
                    ann_return = (row['lastPrice'] / row['strike']) * (365 / max(1, dte)) * 100
                    puffer = ((price/row['strike'])-1)*100
                    
                    with st.expander(f"Strike {row['strike']}$ | {ann_return:.1f}% p.a."):
                        st.write(f"Prämie: **{row['lastPrice']}$**")
                        st.write(f"Sicherheitspuffer: **{puffer:.1f}%**")
                        st.write(f"Abstand zum 60T-Tief: In Analyse...")
            
            # Externe Links für schnelle Analyse
            st.divider()
            c1, c2 = st.columns(2)
            c1.link_button("Onvista Check", f"https://www.onvista.de/suche/?searchTerm={ticker}")
            c2.link_button("Stock3 Chart", f"https://stock3.com/aktien/{ticker}-aktie")

        except Exception as e:
            st.error("Yahoo blockiert die Optionsketten noch. Der Aktienkurs (oben) geht aber wieder!")
    else:
        st.error("Alle Datenquellen sind aktuell gesperrt. Bitte kurz warten.")

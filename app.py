import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# Seite einstellen
st.set_page_config(page_title="Stillhalter Scanner", layout="centered")

# --- CACHING FUNKTION ---
# Diese Funktion merkt sich die Daten für 10 Minuten (600 Sekunden)
@st.cache_data(ttl=600)
def get_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1y")
    return stock, hist

st.title("🛡️ Stillhalter Scanner")

ticker_symbol = st.text_input("Aktien-Ticker (z.B. AAPL)", "AAPL").upper()
max_delta_input = st.slider("Max. Delta (für Puts)", 0.05, 0.40, 0.20, step=0.01)

if ticker_symbol:
    try:
        # Daten über die Cache-Funktion laden
        stock, hist = get_data(ticker_symbol)
        
        if hist.empty:
            st.error("Keine Daten gefunden. Ticker korrekt?")
        else:
            current_price = hist['Close'].iloc[-1]
            sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            support_level = hist['Low'].tail(60).min() # 60-Tage Tief

            # Anzeige der Metriken
            st.metric("Kurs", f"{current_price:.2f} $")
            col1, col2 = st.columns(2)
            col1.write(f"**Trend:** {'✅ Bullish' if current_price > sma_200 else '⚠️ Bearish'}")
            col2.write(f"**Support (60T):** {support_level:.2f} $")

            # Chart für Handy-Ansicht
            st.line_chart(hist['Close'].tail(90))

            # Optionsdaten
            expirations = stock.options
            if expirations:
                expiry = st.selectbox("Ablaufdatum", expirations)
                dte = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
                
                # Chain laden (auch hier könnte man Caching nutzen, falls nötig)
                puts = stock.option_chain(expiry).puts
                
                # Filter: Nur Puts unter aktuellem Kurs
                filtered_puts = puts[puts['strike'] <= current_price].sort_values(by='strike', ascending=False)

                st.subheader(f"Puts für {expiry}")
                for index, row in filtered_puts.head(5).iterrows():
                    # Berechnung Annualized Return
                    ann_return = (row['lastPrice'] / (row['strike'])) * (365 / max(1, dte)) * 100
                    
                    with st.expander(f"Strike {row['strike']}$ - {ann_return:.1f}% p.a."):
                        st.write(f"Prämie: {row['lastPrice']}$")
                        st.write(f"IV: {row['impliedVolatility']:.1%}")
                        st.write(f"Abstand: {((current_price/row['strike'])-1)*100:.1f}%")

    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")
        st.info("Yahoo blockiert gerade. Bitte in 10 Min nochmal probieren oder anderen Ticker testen.")

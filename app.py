# ==========================================================
#  CapTrader AI Market Scanner – OPTION A (REAL MODE INTEGRATED)
#  Vollständige Version – Segment 1/10
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import random
import requests
import time
from datetime import datetime, timedelta
import concurrent.futures
from scipy.stats import norm

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="CapTrader AI Market Scanner",
    layout="wide"
)

# ----------------------------------------------------------
# REAL-MODE: WATCHLIST (NEU)
# ----------------------------------------------------------
def get_combined_watchlist():
    """
    Kombiniert Megacaps, AI/Chips, Growth, Fintech und Crypto
    → Wird für den echten Profi-Scan genutzt
    """
    mega = ["AAPL", "MSFT", "AMZN", "META", "GOOGL"]
    ai = ["NVDA", "AMD", "MU", "AVGO", "SMCI"]
    growth = ["TSLA", "PLTR", "CRWD", "TTD", "PANW"]
    fintech = ["AFRM", "PYPL", "SQ", "HOOD"]
    crypto = ["COIN", "MSTR", "RIOT"]

    return sorted(list(set(mega + ai + growth + fintech + crypto)))

# ----------------------------------------------------------
# REAL-MODE: PIVOT-POINTS (NEU)
# ----------------------------------------------------------
def get_pivot_points(df):
    """
    Berechnet klassische Pivot-Punkte auf Basis der letzten Tageskerze vor heute.
    Wird im Depot-Manager genutzt.
    """
    if df.empty or len(df) < 3:
        return None

    last = df.iloc[-2]
    high = last["High"]
    low = last["Low"]
    close = last["Close"]

    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)

    return {"P": pivot, "R1": r1, "R2": r2, "S1": s1, "S2": s2}

# ----------------------------------------------------------
# REAL-MODE: Earnings (NEU)
# ----------------------------------------------------------
def get_next_earnings(ticker_obj):
    """
    Holt nächstes Earnings-Datum über Yahoo Finance.
    Gibt nur den Tag und Monat zurück (XX.XX.)
    """
    try:
        cal = ticker_obj.calendar
        if cal is None or cal.empty:
            return "---"
        # Index = vollständiges Datum
        dt = cal.index[0]
        return dt.strftime("%d.%m.")
    except:
        return "---"

# ==========================================================
#  Segment 2/10 – RSI, Analysten-Scoring, Market-Context
# ==========================================================

# ----------------------------------------------------------
# RSI (identisch zu deinem Original, aber stabiler)
# ----------------------------------------------------------
def calculate_rsi_vectorized(series, window=14):
    """
    Stabiler RSI mit Fallback bei zu kurzem Datenfenster.
    """
    if series.empty or len(series) < window:
        return pd.Series([50] * len(series))
    
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# ----------------------------------------------------------
# Analysten-Konfidenz (verbessert, robustere Logik)
# ----------------------------------------------------------
def get_analyst_conviction(info, is_demo=False):
    """
    Liefert Analysten-Sentiment + Farbe.
    Robust, auch wenn Yahoo manche Felder nicht mehr liefert.
    """
    if is_demo:
        return "✅ Stark (Ziel: +18%)", "#27ae60"

    try:
        curr = info.get("currentPrice") or info.get("lastPrice")
        target = (
            info.get("targetMedianPrice")
            or info.get("targetMeanPrice")
            or None
        )
        rev_growth = info.get("revenueGrowth", 0)

        # Hyper-Growth Erkennung
        if rev_growth and rev_growth > 0.30:
            return "🚀 HYPER-GROWTH", "#9b59b6"

        # Analysten-Ziele vorhanden?
        if curr and target:
            upside = (target / curr - 1) * 100
            if upside > 15:
                return f"✅ Stark (+{upside:.0f}%)", "#10b981"
            if upside > 5:
                return f"⚖️ Neutral (+{upside:.0f}%)", "#6b7280"

        return "🔍 Check", "#6b7280"

    except:
        return "🔍 Check", "#6b7280"

# ----------------------------------------------------------
# Market-Context (echter Modus + Demo-Modus)
# ----------------------------------------------------------
def get_market_context(is_demo=False):
    """
    Holt Nasdaq, VIX, Bitcoin, RSI & Abstand zum 20er SMA.
    Robust gegen API-Fails.
    """
    if is_demo:
        return {
            "cp": 16540,
            "rsi": 45,
            "dist": -1.2,
            "vix": 18.5,
            "btc": 92000,
            "fg": 62
        }

    result = {
        "cp": 0, "rsi": 50, "dist": 0,
        "vix": 20, "btc": 0, "fg": 50
    }

    try:
        data = yf.download(
            ["^NDX", "^VIX", "BTC-USD"],
            period="60d",
            interval="1d",
            progress=False
        )

        # Nasdaq 100
        if "^NDX" in data["Close"]:
            ndx = data["Close"]["^NDX"].dropna()
            cp = ndx.iloc[-1]
            sma20 = ndx.rolling(20).mean().iloc[-1]
            result["cp"] = cp
            result["dist"] = ((cp - sma20) / sma20) * 100
            result["rsi"] = calculate_rsi_vectorized(ndx).iloc[-1]

        # VIX
        if "^VIX" in data["Close"]:
            result["vix"] = float(data["Close"]["^VIX"].iloc[-1])

        # BTC
        if "BTC-USD" in data["Close"]:
            result["btc"] = float(data["Close"]["BTC-USD"].iloc[-1])

        # Fear & Greed Index (Fallback)
        try:
            fng = requests.get("https://api.alternative.me/fng/")
            result["fg"] = int(fng.json()["data"][0]["value"])
        except:
            result["fg"] = 50

    except:
        pass

    return result

# ==========================================================
#  Segment 3/10 – Cache, Batch-Download, OptionChain (Real-Mode)
# ==========================================================

# ----------------------------------------------------------
# Caching für Yahoo-Batchdaten
# ----------------------------------------------------------
@st.cache_data(ttl=3600)
def get_batch_data_cached(tickers, is_demo=False):
    """
    Holt historische Kursdaten für alle Ticker.
    Im Demo-Modus werden synthetische Daten erzeugt.
    """
    if is_demo:
        st.warning("🚧 Demo-Modus aktiv: Zeige generierte Testdaten.")
        dates = pd.date_range(end=datetime.now(), periods=250)

        multi = pd.MultiIndex.from_product(
            [tickers, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        )

        df = pd.DataFrame(
            np.random.randn(250, len(tickers)*6) * 10 + 150,
            index=dates,
            columns=multi
        )
        return df

    # Echtmodus
    try:
        data = yf.download(
            tickers,
            period="250d",
            group_by="ticker",
            auto_adjust=True,
            progress=False
        )
        return data
    except Exception as e:
        st.error(f"⚠️ Yahoo-Fehler beim Batch-Download: {e}")
        return pd.DataFrame()


# ----------------------------------------------------------
# OptionChain Loader – Real-Mode
# Variante 2: Expiration ≈ 30 Tage (nächste passende wählen)
# ----------------------------------------------------------
def find_expiration_date(ticker_obj, target_days=30):
    """
    Variante 2: Wählt automatisch das Expiry-Datum,
    das der gewünschten Restlaufzeit (~30 Tage) am nächsten kommt.
    """
    try:
        expirations = ticker_obj.options
        if not expirations:
            return None

        today = datetime.now().date()

        best = None
        min_diff = 9999

        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                diff = abs((exp_date - today).days - target_days)

                if diff < min_diff:
                    min_diff = diff
                    best = exp_str
            except:
                continue

        return best

    except:
        return None


def load_option_chain(symbol, is_put=True, target_days=30):
    """
    Lädt die richtige Optionskette (Put/Call) für den echten Modus.
    Nutzt Variante 2: automatische Suche nach ≈30 Tagen Laufzeit.
    """
    try:
        tk = yf.Ticker(symbol)
        exp = find_expiration_date(tk, target_days=target_days)

        if exp is None:
            return None

        chain = tk.option_chain(exp)

        df = chain.puts if is_put else chain.calls

        # Nur relevante Spalten behalten
        keep = ["strike", "bid", "ask", "impliedVolatility", "delta", "openInterest", "volume"]
        df = df[[c for c in keep if c in df.columns]]

        df = df.rename(columns={
            "impliedVolatility": "IV",
            "openInterest": "OI"
        })

        return df

    except Exception as e:
        print("OptionChain Fehler:", e)
        return None


# ----------------------------------------------------------
# Sterne-Logik aus deinem Original (leicht verbessert)
# ----------------------------------------------------------
def get_stars_logic(analyst_label, uptrend):
    """
    Berechnet deine Sterne-Qualität basierend auf Analysten + SMA200.
    """
    score = 1.0

    if "HYPER" in analyst_label:
        score = 3.0
    elif "Stark" in analyst_label:
        score = 2.0

    if uptrend:
        score += 1.0

    return score, "⭐" * int(score)

# ==========================================================
#  Segment 4/10 – Sidebar, Marktampel, Marktmetriken
# ==========================================================

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System-Einstellungen")

    demo_mode = st.toggle(
        "🛠️ Demo-Modus (API Bypass)",
        value=False,
        help="Aktivieren, wenn Yahoo Finance dich temporär blockiert."
    )

    st.markdown("---")
    st.header("🛡️ Scanner-Filter")

    otm_puffer_slider = st.slider("OTM-Puffer (%)", 5, 25, 12)
    min_yield_pa = st.number_input("Mindestrendite p.a. (%)", 5, 100, 12)
    min_stock_price, max_stock_price = st.slider(
        "Preis ($)",
        0, 1000, (40, 600)
    )
    min_mkt_cap = st.slider("Market Cap (Mrd. $)", 1, 2000, 15)
    only_uptrend = st.checkbox("Nur SMA200 Uptrend", value=False)
    test_modus = st.checkbox("🔍 Kleiner Scan (12 Ticker)", value=False)


# ----------------------------------------------------------
# MARKT-AMPER + GLOBALER MARKET CONTEXT
# ----------------------------------------------------------
st.markdown("## 🌍 Globales Markt-Monitoring")

market = get_market_context(is_demo=demo_mode)

# Ampel-Logik
ampel_color = "#27ae60"        # Grün
ampel_text  = "MARKT STABIL"

if market["dist"] < -2 or market["vix"] > 24:
    ampel_color, ampel_text = "#e74c3c", "🚨 MARKT-ALARM"
elif market["rsi"] > 70:
    ampel_color, ampel_text = "#f39c12", "⚠️ ÜBERHITZT"

# HTML-Ampel
st.markdown(f"""
    <div style="
        background-color: {ampel_color};
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;">
        <h2 style="margin:0;">{ampel_text} {'(DEMO)' if demo_mode else ''}</h2>
    </div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# Marktmetriken (4 Spalten)
# ----------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("Nasdaq 100", f"{market['cp']:,.0f}", f"{market['dist']:.1f}%")
c2.metric("Bitcoin", f"{market['btc']:,.0f} $")
c3.metric("VIX (Angst)", f"{market['vix']:.2f}")
c4.metric("Nasdaq RSI", f"{int(market['rsi'])}")

st.markdown("---")

# ==========================================================
#  Segment 5/10 – Profi-Scanner (Demo-Pfad + Echtmodus Basis)
# ==========================================================

# Stelle sicher, dass Session-State existiert
if "profi_scan_results" not in st.session_state:
    st.session_state.profi_scan_results = []


# ----------------------------------------------------------
# BUTTON: Profi-Scan starten
# ----------------------------------------------------------
st.markdown("## 🚀 Profi-Scanner")
if st.button("Profi-Scan starten", use_container_width=True):
    all_results = []

    # ------------------------------------------------------
    # PFAD A – DEMO-MODUS
    # ------------------------------------------------------
    if demo_mode:
        with st.spinner("Generiere Demo-Setups im Original-Design..."):
            time.sleep(1)

            demo_tickers = [
                "NVDA", "TSLA", "AAPL", "AMD", "MSFT", "MU",
                "PLTR", "AMZN", "META", "COIN", "MSTR", "NFLX"
            ]

            for sym in demo_tickers:

                # Uptrend erzwingen falls Filter aktiv
                is_uptrend = True if only_uptrend else random.choice([True, False])
                price = random.uniform(100, 950)

                all_results.append({
                    "symbol": sym,
                    "stars_str": "⭐⭐" + ("⭐" if random.random() > 0.5 else ""),
                    "sent_icon": "🟢" if is_uptrend else "🔹",
                    "status": "Trend" if is_uptrend else "Dip",
                    "y_pa": random.uniform(15.0, 38.0),
                    "strike": price * 0.85,
                    "bid": random.uniform(1.5, 5.0),
                    "puffer": random.uniform(10, 22),
                    "delta": random.uniform(-0.10, -0.35),
                    "em_pct": random.uniform(0.5, 4.5) * (1 if random.random() > 0.5 else -1),
                    "em_safety": random.uniform(0.8, 2.1),
                    "tage": 32,
                    "rsi": random.randint(30, 75),
                    "mkt_cap": random.uniform(50, 2500),
                    "earn": random.choice(["15.03.", "22.04.", "---"]),
                    "analyst_label": random.choice(["Stark", "Kaufen", "Hyper-Growth"]),
                    "analyst_color": random.choice(["#10b981", "#3b82f6", "#8b5cf6"])
                })

        st.session_state.profi_scan_results = all_results


    # ------------------------------------------------------
    # PFAD B – ECHT-MODUS
    # ------------------------------------------------------
    else:
        # Ticker bestimmen
        ticker_liste = (
            ["NVDA", "TSLA", "AMD", "MU"]
            if test_modus else
            get_combined_watchlist()
        )

        with st.spinner(f"Scanne {len(ticker_liste)} Ticker..."):
            batch_data = get_batch_data_cached(ticker_liste, is_demo=False)

            if not isinstance(batch_data, pd.DataFrame):
                st.error("Konnte Batch-Daten nicht laden.")
                st.stop()

            # ----------------------------------------------
            # Worker-Funktion: Echt-Daten eines Symbols prüfen
            # ----------------------------------------------
            def check_stock(sym):
                try:
                    # Einzel-DF extrahieren
                    hist = batch_data[sym] if len(ticker_liste) > 1 else batch_data
                    if hist.empty:
                        return None

                    # Preis + SMA200
                    price = hist["Close"].iloc[-1]
                    sma200 = hist["Close"].rolling(200).mean().iloc[-1]
                    is_uptrend = price > sma200

                    # Uptrend-Filter
                    if only_uptrend and not is_uptrend:
                        return None

                    # RSI
                    rsi_series = calculate_rsi_vectorized(hist["Close"])
                    rsi_current = int(rsi_series.iloc[-1])

                    # Yahoo-Objekt
                    tk = yf.Ticker(sym)
                    info = tk.info

                    # Analystendaten
                    analyst_label, analyst_color = get_analyst_conviction(info)

                    # Earnings
                    earn_date = get_next_earnings(tk)

                    # Market Cap
                    mkt_cap = info.get("marketCap", 0) / 1e9

                    # 🔜 OPTIONSDATEN KOMMEN IN SEGMENT 6
                    # Hier kommen erst Dummy-Werte – werden gleich ersetzt
                    return {
                        "symbol": sym,
                        "stars_str": "⭐⭐⭐",  # wird später ausgebaut
                        "sent_icon": "🟢" if is_uptrend else "🔹",
                        "status": "Trend" if is_uptrend else "Dip",
                        "y_pa": 0.0,         # wird in Segment 6 echt berechnet
                        "strike": 0.0,       # Segment 6
                        "bid": 0.0,          # Segment 6
                        "puffer": 0.0,       # Segment 6
                        "delta": 0.0,        # Segment 6
                        "em_pct": 0.0,       # Segment 6
                        "em_safety": 0.0,    # Segment 6
                        "tage": 30,
                        "rsi": rsi_current,
                        "mkt_cap": mkt_cap,
                        "earn": earn_date,
                        "analyst_label": analyst_label,
                        "analyst_color": analyst_color
                    }

                except Exception as e:
                    print(f"Fehler bei {sym}: {e}")
                    return None

            # ----------------------------------------------
            # Thread-Pool: Parallel Scan
            # ----------------------------------------------
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exe:
                futures = [exe.submit(check_stock, s) for s in ticker_liste]
                for f in concurrent.futures.as_completed(futures):
                    result = f.result()
                    if result:
                        results.append(result)

            st.session_state.profi_scan_results = results

# ==========================================================
#  Segment 6/10 – Real-Mode Optionslogik + Ergebnis-Patching
# ==========================================================

# Nur wenn wir im Echtmodus sind: Ergebnisse ergänzen
if not demo_mode and "profi_scan_results" in st.session_state:

    enhanced_results = []

    for item in st.session_state.profi_scan_results:

        sym = item["symbol"]
        price = None

        try:
            # Kurs aus Yahoo holen
            tk = yf.Ticker(sym)
            hist = tk.history(period="1y")

            if not hist.empty:
                price = hist["Close"].iloc[-1]
            else:
                enhanced_results.append(item)
                continue
        except:
            enhanced_results.append(item)
            continue

        # ------------------------------------------------------
        # OPTIONSDATEN LADEN (Put, 30 Tage Restlaufzeit)
        # ------------------------------------------------------
        opt_df = load_option_chain(sym, is_put=True, target_days=30)

        if opt_df is None or opt_df.empty:
            # Wenn keine Optionsdaten → Eintrag unverändert lassen
            enhanced_results.append(item)
            continue

        # ------------------------------------------------------
        # PASSENDEN Strike suchen (85% von Spot)
        # ------------------------------------------------------
        target_strike = price * 0.85

        # Finde Option mit minimaler Distanz zum Zielstrike
        row = opt_df.iloc[(opt_df["strike"] - target_strike).abs().argsort()[0]]

        strike = float(row["strike"])
        bid = float(row.get("bid", 0) or 0)
        ask = float(row.get("ask", 0) or 0)
        delta = float(row.get("delta", 0) or 0)
        iv = float(row.get("IV", 0) or 0)

        # ------------------------------------------------------
        # BERECHNUNGEN
        # ------------------------------------------------------

        # Abstand OTM (Puffer)
        puffer_pct = ((strike / price) - 1) * 100

        # Laufzeit = echte Differenz
        exp = find_expiration_date(tk, target_days=30)
        if exp:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            today = datetime.now().date()
            days_to_exp = max(1, (exp_date - today).days)
        else:
            days_to_exp = 30

        # Yield p.a. (standardisierte Formel)
        try:
            y_pa = (bid / strike) * (365 / days_to_exp) * 100
        except:
            y_pa = 0.0

        # Expected Move (EM %)
        # IV wird von Yahoo als Dezimalzahl geliefert (0.40 = 40%)
        try:
            em_pct = iv * 100 * np.sqrt(days_to_exp / 365)
        except:
            em_pct = 0.0

        # Safety Faktor
        try:
            em_safety = abs(puffer_pct / em_pct) if em_pct > 0 else 0
        except:
            em_safety = 0.0

        # ------------------------------------------------------
        # STERNE-NEUBERECHNUNG (Analysten + Trend)
        # ------------------------------------------------------
        uptrend = item["status"] == "Trend"
        stars_val, stars_str = get_stars_logic(item["analyst_label"], uptrend)

        # ------------------------------------------------------
        # ERGEBNISSE PATCHEN
        # ------------------------------------------------------
        item.update({
            "strike": strike,
            "bid": bid,
            "delta": delta,
            "puffer": puffer_pct,
            "y_pa": y_pa,
            "em_pct": em_pct,
            "em_safety": em_safety,
            "stars_str": stars_str
        })

        enhanced_results.append(item)

    st.session_state.profi_scan_results = enhanced_results

# ==========================================================
#  Segment 7/10 – Profi-Scanner HTML Rendering (FULL CARDS)
# ==========================================================

# ----------------------------------------------------------
# Anzeige der Ergebnisse (HTML Cards)
# ----------------------------------------------------------
if "profi_scan_results" in st.session_state and st.session_state.profi_scan_results:

    results = st.session_state.profi_scan_results

    st.subheader(f"🎯 Top-Setups nach Qualität ({len(results)} Treffer)")
    cols = st.columns(4)
    now_dt = datetime.now()

    for idx, res in enumerate(results):

        with cols[idx % 4]:

            # Grundlagen
            symbol = res["symbol"]
            stars = res.get("stars_str", "⭐")
            status_txt = res.get("status", "Trend")
            sent_icon = res.get("sent_icon", "🟢")
            analyst_label = res.get("analyst_label", "Keine Analyse")
            analyst_color = res.get("analyst_color", "#888888")
            earn_str = res.get("earn", "---")

            # Werte
            y_pa = res.get("y_pa", 0.0)
            strike = res.get("strike", 0.0)
            bid = res.get("bid", 0.0)
            puffer = res.get("puffer", 0.0)
            delta_val = abs(res.get("delta", 0.0))
            em_pct = res.get("em_pct", 0.0)
            em_safety = res.get("em_safety", 0.0)
            tage = res.get("tage", 30)
            rsi_val = int(res.get("rsi", 50))
            mkt_cap = res.get("mkt_cap", 0)

            # ------------------------------------------------------
            # Farb- & Bewertungslogiken
            # ------------------------------------------------------

            # Statusfarbe (Trend / Dip)
            status_color = "#10b981" if status_txt == "Trend" else "#3b82f6"

            # RSI Farbe
            if rsi_val >= 70:
                rsi_style = "color: #ef4444; font-weight: 900;"
            elif rsi_val <= 35:
                rsi_style = "color: #10b981; font-weight: 700;"
            else:
                rsi_style = "color: #4b5563; font-weight: 700;"

            # Delta Farbe
            delta_color = (
                "#10b981" if delta_val < 0.20 else
                "#f59e0b" if delta_val < 0.30 else
                "#ef4444"
            )

            # EM Farbe
            em_color = (
                "#10b981" if em_safety >= 1.5 else
                "#f59e0b" if em_safety >= 1.0 else
                "#ef4444"
            )

            # Earnings Risiko (roter Rahmen)
            is_earning_risk = False
            if earn_str and earn_str != "---":
                try:
                    earn_date = datetime.strptime(f"{earn_str}{now_dt.year}", "%d.%m.%Y")
                    days_to_earn = (earn_date - now_dt).days
                    if 0 <= days_to_earn <= tage:
                        is_earning_risk = True
                except:
                    pass

            # Kartenstil
            if is_earning_risk:
                card_border = "4px solid #ef4444"
                card_shadow = "0 8px 16px rgba(239, 68, 68, 0.25)"
                card_bg = "#fff6f6"
            else:
                card_border = "1px solid #e5e7eb"
                card_shadow = "0 4px 6px -1px rgba(0,0,0,0.05)"
                card_bg = "#ffffff"

            # ------------------------------------------------------
            # HTML Card
            # ------------------------------------------------------
            html_code = f"""
<div style="
    background: {card_bg};
    border: {card_border};
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 20px;
    box-shadow: {card_shadow};
    font-family: sans-serif;
">
    <!-- HEADER -->
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
        <span style="font-size:1.2em; font-weight:800; color:#111827;">
            {symbol} <span style="color:#f59e0b; font-size:0.8em;">{stars}</span>
        </span>
        <span style="
            font-size:0.75em; font-weight:700; color:{status_color};
            background:{status_color}10; padding:2px 8px; border-radius:6px;">
            {sent_icon} {status_txt}
        </span>
    </div>

    <!-- YIELD -->
    <div style="margin:10px 0;">
        <div style="font-size:0.7em; color:#6b7280; font-weight:600; text-transform:uppercase;">Yield p.a.</div>
        <div style="font-size:1.9em; font-weight:900; color:#111827;">{y_pa:.1f}%</div>
    </div>

    <!-- GRID MIT STRIKE / BID / PUFFER / DELTA -->
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px;">

        <div style="border-left:3px solid #8b5cf6; padding-left:8px;">
            <div style="font-size:0.6em; color:#6b7280;">Strike</div>
            <div style="font-size:0.9em; font-weight:700;">{strike:.1f}$</div>
        </div>

        <div style="border-left:3px solid #f59e0b; padding-left:8px;">
            <div style="font-size:0.6em; color:#6b7280;">Bid</div>
            <div style="font-size:0.9em; font-weight:700;">{bid:.2f}$</div>
        </div>

        <div style="border-left:3px solid #3b82f6; padding-left:8px;">
            <div style="font-size:0.6em; color:#6b7280;">Puffer</div>
            <div style="font-size:0.9em; font-weight:700;">{puffer:.1f}%</div>
        </div>

        <div style="border-left:3px solid {delta_color}; padding-left:8px;">
            <div style="font-size:0.6em; color:#6b7280;">Delta</div>
            <div style="font-size:0.9em; font-weight:700; color:{delta_color};">{delta_val:.2f}</div>
        </div>

    </div>

    <!-- EM / SECURITY BOX -->
    <div style="
        background:{em_color}10; padding:6px 10px;
        border-radius:8px; margin-bottom:12px;
        border:1px dashed {em_color};
    ">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.65em; color:#4b5563; font-weight:bold;">Expected Move (EM):</span>
            <span style="font-size:0.75em; font-weight:800; color:{em_color};">{em_pct:.1f}%</span>
        </div>
        <div style="font-size:0.6em; color:#6b7280; margin-top:2px;">
            Sicherheit: <b>{em_safety:.1f}x</b>
        </div>
    </div>

    <hr style="border:0; border-top:1px solid #f3f4f6; margin:10px 0;">

    <!-- RSI / MKT CAP / EARNINGS -->
    <div style="
        display:flex; justify-content:space-between; align-items:center;
        font-size:0.72em; color:#4b5563; margin-bottom:10px;
    ">
        <span>⏳ <b>{tage}d</b></span>

        <div style="display:flex; gap:4px;">
            <span style="background:#f3f4f6; padding:2px 6px; border-radius:4px; {rsi_style}">
                RSI: {rsi_val}
            </span>

            <span style="background:#f3f4f6; padding:2px 6px; border-radius:4px; font-weight:700;">
                {mkt_cap:.0f}B
            </span>
        </div>

        <span style="font-weight:800; color:{'#ef4444' if is_earning_risk else '#6b7280'};">
            {'⚠️' if is_earning_risk else '🗓️'} {earn_str}
        </span>
    </div>

    <!-- ANALYSTEN BOX -->
    <div style="
        background:{analyst_color}15; color:{analyst_color};
        padding:8px; border-radius:8px;
        border-left:4px solid {analyst_color};
        font-size:0.7em; font-weight:bold; text-align:center;
    ">
        🚀 {analyst_label}
    </div>
</div>
"""

            st.markdown(html_code, unsafe_allow_html=True)

# ==========================================================
#  Segment 8/10 – DEPOT-MANAGER (inkl. Pivot Points + Sterne)
# ==========================================================

st.markdown("---")
st.header("🛠️ Depot-Manager: Bestandsverwaltung & Reparatur")

# Initialer State
if "depot_data_cache" not in st.session_state:
    st.session_state.depot_data_cache = None


# ----------------------------------------------------------
# Depot-Analyse starten (Button)
# ----------------------------------------------------------
if st.session_state.depot_data_cache is None:

    st.info("📦 Die Depot-Analyse ist aktuell pausiert.")

    if st.button("🚀 Depot jetzt analysieren (inkl. Sterne-Check)", use_container_width=True):

        with st.spinner("Analysiere Qualität, RSI & Pivot-Signale..."):

            # Dein Original-Portfolio
            my_assets = {
                "LRCX": [100, 210], "MU": [100, 390], "AFRM": [100, 76.00],
                "ELF": [100, 109.00], "ETSY": [100, 67.00], "GTLB": [100, 41.00],
                "GTM": [100, 17.00], "HIMS": [100, 36.00], "HOOD": [100, 120.00],
                "JKS": [100, 50.00], "NVO": [100, 97.00], "RBRK": [100, 70.00],
                "SE": [100, 170.00], "TTD": [100, 102.00]
            }

            ticker_keys = list(my_assets.keys())

            # Batch‑Download für Performance
            depot_batch = yf.download(
                ticker_keys,
                period="250d",
                group_by="ticker",
                auto_adjust=True,
                progress=False
            )

            depot_results = []

            # ------------------------------------------------------
            # Jedes Depot‑Asset einzeln evaluieren
            # ------------------------------------------------------
            for symbol in ticker_keys:
                try:
                    hist = depot_batch[symbol]
                    if hist.empty:
                        continue

                    # Preis, Performance
                    price = hist["Close"].iloc[-1]
                    qty, entry = my_assets[symbol]
                    perf_pct = ((price - entry) / entry) * 100

                    # Trend (SMA200)
                    sma200 = hist["Close"].rolling(200).mean().iloc[-1]
                    uptrend = price > sma200

                    # RSI
                    rsi_series = calculate_rsi_vectorized(hist["Close"])
                    rsi_val = int(rsi_series.iloc[-1])

                    # Pivot Points (klassisch)
                    piv = get_pivot_points(hist)
                    s2_support = piv["S2"] if piv else None
                    r2_target = piv["R2"] if piv else None

                    # Yahoo‑Info für Analysten‑Daten
                    tk = yf.Ticker(symbol)
                    info = tk.info
                    analyst_label, _ = get_analyst_conviction(info)

                    # Sterne‑Logik
                    stars_num = 1
                    if "HYPER" in analyst_label:
                        stars_num = 3
                    elif "Stark" in analyst_label:
                        stars_num = 2

                    if uptrend:
                        stars_num += 0.5  # Bonus bei Trend

                    stars_str = "⭐" * int(stars_num)

                    # Reparatur‑Signal (Put)
                    repair_signal = "⏳ Warten"
                    if rsi_val < 35:
                        repair_signal = "🟢 JETZT (RSI < 35)"
                    if s2_support and price <= s2_support * 1.02:
                        repair_signal = "🟢 JETZT (S2‑Touch)"

                    depot_results.append({
                        "Ticker": f"{symbol} {stars_str}",
                        "Einstand": f"{entry:.2f} $",
                        "Aktuell": f"{price:.2f} $",
                        "P/L %": f"{perf_pct:+.1f}%",
                        "RSI": rsi_val,
                        "Repair (Put)": repair_signal,
                        "S2 Support": f"{s2_support:.2f} $" if s2_support else "---",
                        "R2 Ziel": f"{r2_target:.2f} $" if r2_target else "---"
                    })

                except Exception as e:
                    print(f"Depot-Fehler bei {symbol}: {e}")
                    continue

            st.session_state.depot_data_cache = depot_results
            st.rerun()


# ----------------------------------------------------------
# DEPOT-TABELLE anzeigen
# ----------------------------------------------------------
else:
    st.dataframe(
        pd.DataFrame(st.session_state.depot_data_cache),
        use_container_width=True,
        hide_index=True
    )

    if st.button("🔄 Depot-Daten aktualisieren"):
        st.session_state.depot_data_cache = None
        st.rerun()

# ==========================================================
#  Segment 9/10 – Profi-Analyse & Trading-Cockpit (Real Mode)
# ==========================================================

st.markdown("---")
st.markdown("## 🔍 Profi-Analyse & Trading-Cockpit")

# ----------------------------------------------------------
# Eingabe-Feld
# ----------------------------------------------------------
c_input1, _ = st.columns([1, 2])
with c_input1:
    symbol_input = st.text_input("Ticker Symbol", value="MU").upper()


if symbol_input:

    # Basisstruktur für Ergebnis
    cockpit = {
        "symbol": symbol_input,
        "status": "Standby",
        "sent_icon": "⚪",
        "stars": "⭐⭐",
        "strike": 0,
        "bid": 0,
        "puffer": 0,
        "delta": 0,
        "y_pa": 0,
        "em_pct": 0,
        "em_safety": 0,
        "tage": 30,
        "rsi": 50,
        "mkt_cap": 0,
        "earn": "---",
        "analyst_label": "Lade Daten...",
        "analyst_color": "#6b7280"
    }

    with st.spinner(f"Analysiere {symbol_input}..."):

        if demo_mode:
            # ------------------------------
            # DEMO-VERSION DES COCKPITS
            # ------------------------------
            cockpit.update({
                "status": "Trend",
                "sent_icon": "🟢",
                "stars": "⭐⭐⭐",
                "strike": 85.0,
                "bid": 2.45,
                "puffer": 15.2,
                "delta": -0.18,
                "y_pa": 28.4,
                "em_pct": 3.2,
                "em_safety": 1.4,
                "rsi": 42,
                "mkt_cap": 145,
                "earn": "18.03.",
                "analyst_label": "🚀 HYPER-GROWTH",
                "analyst_color": "#9b59b6"
            })

        else:
            # ------------------------------
            # REALE DATEN HOLEN
            # ------------------------------
            try:
                tk = yf.Ticker(symbol_input)
                hist = tk.history(period="1y")

                if not hist.empty:

                    # Spot Price
                    cp = hist["Close"].iloc[-1]

                    # Trend via SMA200
                    sma200 = hist["Close"].rolling(200).mean().iloc[-1]
                    uptrend = cp > sma200
                    cockpit["status"] = "Trend" if uptrend else "Dip"
                    cockpit["sent_icon"] = "🟢" if uptrend else "🔹"

                    # RSI
                    rsi_series = calculate_rsi_vectorized(hist["Close"])
                    cockpit["rsi"] = int(rsi_series.iloc[-1])

                    # Market Cap
                    cockpit["mkt_cap"] = tk.info.get("marketCap", 0) / 1e9

                    # Earnings
                    cockpit["earn"] = get_next_earnings(tk)

                    # Analysten
                    al, col = get_analyst_conviction(tk.info)
                    cockpit["analyst_label"] = al
                    cockpit["analyst_color"] = col

                    # Sterne
                    stars_val, stars_str = get_stars_logic(al, uptrend)
                    cockpit["stars"] = stars_str

                    # ------------------------------------------------------
                    # OPTIONSDATEN (REAL MODE)
                    # ------------------------------------------------------
                    opt = load_option_chain(symbol_input, is_put=True, target_days=30)

                    if opt is not None and not opt.empty:

                        target = cp * 0.85
                        row = opt.iloc[(opt["strike"] - target).abs().argsort()[0]]

                        strike = float(row["strike"])
                        bid = float(row.get("bid", 0) or 0)
                        delta = float(row.get("delta", 0) or 0)
                        iv = float(row.get("IV", 0) or 0)

                        # Laufzeit berechnen
                        exp = find_expiration_date(tk)
                        if exp:
                            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                            today = datetime.now().date()
                            days_to_exp = max(1, (exp_date - today).days)
                        else:
                            days_to_exp = 30

                        # Finanzmathematische Berechnungen
                        puffer_pct = ((strike / cp) - 1) * 100
                        y_pa = (bid / strike) * (365 / days_to_exp) * 100 if strike > 0 else 0
                        em_pct = (iv * 100) * np.sqrt(days_to_exp / 365)
                        em_safety = abs(puffer_pct / em_pct) if em_pct > 0 else 0

                        # Update ins Cockpit
                        cockpit.update({
                            "strike": strike,
                            "bid": bid,
                            "delta": delta,
                            "puffer": puffer_pct,
                            "y_pa": y_pa,
                            "em_pct": em_pct,
                            "em_safety": em_safety
                        })

            except Exception as e:
                st.error(f"Fehler beim Laden der Echt-Daten: {e}")

    # ------------------------------------------------------
    # HTML COCKPIT (Original-Optik behalten)
    # ------------------------------------------------------

    s_color = "#10b981" if cockpit["status"] == "Trend" else "#3b82f6"

    # RSI Farbe
    rsi = cockpit["rsi"]
    if rsi >= 70:
        rsi_style = "color:#ef4444; font-weight:900;"
    elif rsi <= 35:
        rsi_style = "color:#10b981; font-weight:700;"
    else:
        rsi_style = "color:#4b5563; font-weight:700;"

    # Delta Farbe
    delta_val = abs(cockpit["delta"])
    delta_color = "#10b981" if delta_val < 0.20 else "#f59e0b" if delta_val < 0.30 else "#ef4444"

    # EM Farbe
    em_safety = cockpit["em_safety"]
    em_color = "#10b981" if em_safety >= 1.5 else "#f59e0b" if em_safety >= 1.0 else "#ef4444"

    # HTML anzeigen
    st.markdown(f"""
<div style="background:white; border:1px solid #e5e7eb; border-radius:20px;
            padding:25px; box-shadow:0 10px 15px -3px rgba(0,0,0,0.10);
            max-width:800px; margin:auto; font-family:sans-serif;">

    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
        <span style="font-size:2em; font-weight:900; color:#111827;">
            {cockpit["symbol"]} <span style="color:#f59e0b; font-size:0.6em;">{cockpit["stars"]}</span>
        </span>
        <span style="font-size:1em; font-weight:700; color:{s_color};
                     background:{s_color}10; padding:5px 15px; border-radius:10px;">
            {cockpit["sent_icon"]} {cockpit["status"]}
        </span>
    </div>

    <div style="margin:20px 0; text-align:center; background:#f8fafc;
                padding:15px; border-radius:15px;">
        <div style="font-size:0.9em; color:#6b7280; font-weight:600;
                    text-transform:uppercase;">
            Yield p.a.
        </div>
        <div style="font-size:3.5em; font-weight:950; color:#111827;">
            {cockpit["y_pa"]:.1f}%
        </div>
    </div>

    <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr;
                gap:15px; margin-bottom:20px;">

        <div style="border-left:4px solid #8b5cf6; padding-left:12px;">
            <div style="font-size:0.7em; color:#6b7280;">Strike</div>
            <div style="font-size:1.1em; font-weight:800;">{cockpit["strike"]:.1f}$</div>
        </div>

        <div style="border-left:4px solid #f59e0b; padding-left:12px;">
            <div style="font-size:0.7em; color:#6b7280;">Bid</div>
            <div style="font-weight:800; font-size:1.1em;">{cockpit["bid"]:.2f}$</div>
        </div>

        <div style="border-left:4px solid #3b82f6; padding-left:12px;">
            <div style="font-size:0.7em; color:#6b7280;">Puffer</div>
            <div style="font-weight:800; font-size:1.1em;">{cockpit["puffer"]:.1f}%</div>
        </div>

        <div style="border-left:4px solid {delta_color}; padding-left:12px;">
            <div style="font-size:0.7em; color:#6b7280;">Delta</div>
            <div style="font-weight:800; font-size:1.1em; color:{delta_color};">
                {delta_val:.2f}
            </div>
        </div>

    </div>

    <div style="background:{em_color}08; padding:12px; border-radius:12px;
                margin-bottom:20px; border:1px solid {em_color}33;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.85em; color:#4b5563; font-weight:bold;">
                Expected Move (EM):
            </span>
            <span style="font-size:1.1em; font-weight:900; color:{em_color};">
                {cockpit["em_pct"]:.1f}%
            </span>
        </div>
        <div style="font-size:0.75em; color:#6b7280; margin-top:5px;">
            Sicherheit: <b>{cockpit["em_safety"]:.1f}x</b>
        </div>
    </div>

    <div style="display:flex; justify-content:space-around; background:#f3f4f6;
                padding:12px; border-radius:12px; font-size:0.85em; margin-bottom:15px;">
        <span>⏳ {cockpit["tage"]}d</span>
        <span style="{rsi_style}">RSI: {cockpit["rsi"]}</span>
        <span>💎 {cockpit["mkt_cap"]:.0f}B</span>
        <span>🗓️ {cockpit["earn"]}</span>
    </div>

    <div style="background:{cockpit['analyst_color']}15; color:{cockpit['analyst_color']};
                padding:10px; border-radius:10px; border-left:6px solid {cockpit['analyst_color']};
                font-weight:800; text-align:center;">
        {cockpit["analyst_label"]}
    </div>

</div>
    """, unsafe_allow_html=True)

# ==========================================================
#  Segment 10/10 – Optionskette (Put/Call) + Finale
# ==========================================================

st.markdown("---")
st.markdown("## 🧮 Optionskette – Short Put / Short Call")

opt_type = st.radio(
    "Strategie wählen:",
    ["🟢 Short Put (Bullish/Neutral)", "🔴 Short Call (Bearish)"],
    horizontal=True
)

if demo_mode:
    # ------------------------------------------------------
    # DEMO-OPTIONSKETTE
    # ------------------------------------------------------
    st.info("Demo-Optionkette aktiv – synthetische Daten werden angezeigt.")

    data = []
    base_price = 100.0

    is_put = "Put" in opt_type

    for i in range(1, 11):
        # Puffer % OTM
        puffer = -(i * 2.0) if is_put else +(i * 2.0)
        strike = round(base_price * (1 + puffer / 100), 1)

        bid = round(random.uniform(0.5, 4.0) / (i * 0.5 + 1), 2)
        y_pa = round((bid / strike) * (365 / 30) * 100, 1)
        delta = round(0.30 - (i * 0.025), 2)

        data.append({
            "Strike": strike,
            "Bid": bid,
            "Puffer %": puffer,
            "Yield p.a. %": y_pa,
            "Delta": max(delta, 0.01)
        })

    df_demo = pd.DataFrame(data)

    st.dataframe(
        df_demo,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Strike": st.column_config.NumberColumn(format="%.1f $"),
            "Yield p.a. %": st.column_config.NumberColumn(format="%.1f%%"),
            "Puffer %": st.column_config.ProgressColumn(
                "Puffer %",
                format="%.1f%%",
                min_value=-25 if is_put else 0,
                max_value=0 if is_put else 25
            ),
            "Delta": st.column_config.NumberColumn(format="%.2f")
        }
    )

else:
    # ------------------------------------------------------
    # ECHT-OPTIONSKETTE
    # ------------------------------------------------------
    st.info(f"Echte Optionskette für {symbol_input} wird geladen...")

    try:
        tk = yf.Ticker(symbol_input)
        opt_df = load_option_chain(symbol_input, is_put="Put" in opt_type, target_days=30)

        if opt_df is None or opt_df.empty:
            st.error("⚠️ Keine Optionsdaten gefunden.")
        else:

            # Sortieren nach Strike
            opt_df = opt_df.sort_values("strike").copy()

            # Berechnung von Yield & Puffer
            cp = tk.history(period="1d")["Close"].iloc[-1]
            today = datetime.now().date()
            exp = find_expiration_date(tk)
            if exp:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                days_to_exp = max(1, (exp_date - today).days)
            else:
                days_to_exp = 30

            # Zusätzliche Berechnungen
            opt_df["Puffer %"] = (opt_df["strike"] / cp - 1) * 100
            opt_df["Yield p.a. %"] = (opt_df["bid"] / opt_df["strike"]) * (365 / days_to_exp) * 100
            opt_df["Delta"] = opt_df["delta"]

            st.dataframe(
                opt_df[["strike", "bid", "Puffer %", "Yield p.a. %", "Delta", "IV", "OI", "volume"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "strike": st.column_config.NumberColumn("Strike", format="%.2f $"),
                    "bid": st.column_config.NumberColumn("Bid", format="%.2f $"),
                    "Puffer %": st.column_config.ProgressColumn(
                        "Puffer %",
                        help="Abstand zum aktuellen Kurs",
                        format="%.1f%%",
                        min_value=-40.0,
                        max_value=40.0
                    ),
                    "Yield p.a. %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Delta": st.column_config.NumberColumn(format="%.2f"),
                    "IV": st.column_config.NumberColumn("IV", format="%.2f"),
                    "OI": st.column_config.NumberColumn("OI"),
                    "volume": st.column_config.NumberColumn("Vol")
                }
            )

    except Exception as e:
        st.error(f"Fehler beim Laden der echten Optionskette: {e}")

# ----------------------------------------------------------
#  END OF APPLICATION
# ----------------------------------------------------------

st.markdown("---")
st.success("🎉 Die vollständige Option-A Version wurde erfolgreich geladen!")
``

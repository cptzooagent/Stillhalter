 --- SEKTION 2: DEPOT-MANAGER (STABILISIERTE VERSION) ---
st.markdown("---")
st.header("🛠️ Depot-Manager: Bestandsverwaltung & Reparatur")

my_assets = {
    "LRCX": [100, 210], "MU": [100, 390], "AFRM": [100, 76.00], "ELF": [100, 109.00], "ETSY": [100, 67.00],
    "GTLB": [100, 41.00], "GTM": [100, 17.00], "HIMS": [100, 36.00],
    "HOOD": [100, 120.00], "JKS": [100, 50.00], "NVO": [100, 97.00],
    "RBRK": [100, 70.00], "SE": [100, 170.00], "TTD": [100, 102.00]
}

with st.expander("📂 Mein Depot & Strategie-Signale", expanded=True):
    depot_list = []
    for symbol, data in my_assets.items():
        try:
            res = get_stock_data_full(symbol)
            if res[0] is None: continue
            
            # 1. Daten entpacken
            price, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
            qty, entry = data[0], data[1]
            perf_pct = ((price - entry) / entry) * 100

            # 2. KI-Stimmung (OpenClaw) - NUR EINMAL AUFRUFEN
            ki_status, ki_text, _ = get_openclaw_analysis(symbol)
            ki_icon = "🟢" if ki_status == "Bullish" else "🔴" if ki_status == "Bearish" else "🟡"

            # 3. Sterne-Rating (Optimiert mit Try-Except für Speed)
            try:
                # Wir holen info nur einmal
                info_temp = yf.Ticker(symbol).info
                analyst_txt_temp, _ = get_analyst_conviction(info_temp)
                # Kompakte Sterne-Zuweisung
                stars_count = 3 if "HYPER" in analyst_txt_temp else 2 if "Stark" in analyst_txt_temp else 1
                star_display = "⭐" * stars_count
            except:
                star_display = "⭐" # Fallback auf 1 Stern bei Fehler

            # 4. Pivot-Werte sicher extrahieren (ACHTUNG: Unterstrich nutzen!)
            r2_d = pivots.get('R2') if pivots else None
            r2_w = pivots.get('W_R2') if pivots else None
            s2_d = pivots.get('S2') if pivots else None
            s2_w = pivots.get('W_S2') if pivots else None
            
            # --- Hier folgt deine bestehende Put/Call Action Logik ---
            # ... (put_action = ..., call_action = ...)
            
            # Reparatur-Logik (Put)
            put_action = "⏳ Warten"
            if rsi < 35 or (s2_d and price <= s2_d * 1.02):
                put_action = "🟢 JETZT (S2/RSI)"
            if s2_w and price <= s2_w * 1.01:
                put_action = "🔥 EXTREM (Weekly S2)"
            
            # Covered Call Logik (NUR GRÜN WENN R2 EXISTIERT UND > 0 IST)
            call_action = "⏳ Warten"
            if rsi > 55:
                if r2_d and price >= r2_d * 0.98:
                    call_action = "🟢 JETZT (R2/RSI)"
                else:
                    call_action = "🟡 RSI HOCH (Warte auf R2)"

            depot_list.append({
                "Ticker": f"{symbol} {star_display}",
                "Earnings": earn if earn else "---",
                "Einstand": f"{entry:.2f} $",
                "Aktuell": f"{price:.2f} $",
                "P/L %": f"{perf_pct:+.1f}%",
                "KI-Check": f"{ki_icon} {ki_status}",
                "RSI": int(rsi),
                "Short Put (Repair)": put_action,
                "Covered Call": call_action,
                "S2 Daily": f"{s2_d:.2f} $" if s2_d else "---",
                "S2 Weekly": f"{s2_w:.2f} $" if s2_w else "---", # JETZT DABEI
                "R2 Daily": f"{r2_d:.2f} $" if r2_d else "---",
                "R2 Weekly": f"{r2_w:.2f} $" if r2_w else "---"  # JETZT DABEI
            })
        except: continue
    
    if depot_list:
        st.table(pd.DataFrame(depot_list))
        
st.info("💡 **Strategie:** Wenn 'Short Put' auf 🔥 steht, ist die Aktie am wöchentlichen Tiefstand – technisch das sicherste Level zum Verbilligen.")

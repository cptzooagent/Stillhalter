# 4. OPTIONEN TABELLE
                st.markdown("### 🎯 Option-Chain Auswahl")
                heute = datetime.now()
                # Flexibles Fenster: 5 bis 35 Tage
                valid_dates = [d for d in dates if 5 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 35]
                
                if valid_dates:
                    target_date = st.selectbox("📅 Wähle deinen Verfallstag", valid_dates)
                    days_to_expiry = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                    
                    # Buttons zur Auswahl des Typs
                    col_b1, col_b2 = st.columns(2)
                    if "opt_type" not in st.session_state:
                        st.session_state.opt_type = "Puts"
                    
                    if col_b1.button("📉 Short Put (Puts)", use_container_width=True):
                        st.session_state.opt_type = "Puts"
                    if col_b2.button("📈 Short Call (Calls)", use_container_width=True):
                        st.session_state.opt_type = "Calls"

                    # Daten laden basierend auf Auswahl
                    opts_raw = tk.option_chain(target_date)
                    chain = opts_raw.puts if st.session_state.opt_type == "Puts" else opts_raw.calls
                    
                    chain['strike'] = chain['strike'].astype(float)
                    
                    if st.session_state.opt_type == "Puts":
                        chain['Puffer %'] = ((price - chain['strike']) / price) * 100
                        # Nur OTM Puts anzeigen
                        df_disp = chain[(chain['strike'] < price) & (chain['Puffer %'] < 25)].copy()
                    else:
                        chain['Puffer %'] = ((chain['strike'] - price) / price) * 100
                        # Nur OTM Calls anzeigen
                        df_disp = chain[(chain['strike'] > price) & (chain['Puffer %'] < 25)].copy()

                    chain['Yield p.a. %'] = (chain['bid'] / chain['strike']) * (365 / max(1, days_to_expiry)) * 100
                    df_disp = df_disp.sort_values('strike', ascending=(st.session_state.opt_type == "Calls"))

                    def style_rows(row):
                        p = row['Puffer %']
                        if p >= 12: return ['background-color: rgba(39, 174, 96, 0.1)'] * len(row)
                        elif 8 <= p < 12: return ['background-color: rgba(241, 196, 15, 0.1)'] * len(row)
                        return ['background-color: rgba(231, 76, 60, 0.1)'] * len(row)

                    st.subheader(f"Verfügbare {st.session_state.opt_type}")
                    styled_df = df_disp[['strike', 'bid', 'ask', 'Puffer %', 'Yield p.a. %']].style.apply(style_rows, axis=1).format({
                        'strike': '{:.2f} $', 'bid': '{:.2f} $', 'ask': '{:.2f} $',
                        'Puffer %': '{:.1f} %', 'Yield p.a. %': '{:.1f} %'
                    })
                    
                    st.dataframe(styled_df, use_container_width=True, height=450)
                    st.caption(f"🟢 >12% Puffer zum Kurs | 🟡 8-12% | 🔴 <8% (Typ: {st.session_state.opt_type})")

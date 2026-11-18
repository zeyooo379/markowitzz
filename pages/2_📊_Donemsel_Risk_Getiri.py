"""
2_📊_Donemsel_Risk_Getiri: Dönemsel ve Yıllık Kovaryans / Getiri Analizi
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.constants import BIST_HISSELER
from utils.portfolio_engine import calculate_returns, get_market_parameters

st.set_page_config(page_title="Risk & Kovaryans — Markowitzz", page_icon="📊", layout="wide")

st.title("📊 Dönemsel vs. Yıllık Kovaryans ve Risk Matrisi")

prices = st.session_state.get("prices")

if prices is None or prices.empty:
    st.warning("⚠️ Lütfen önce **📥 1_Veri_Cekme** sayfasına giderek hisse senedi verilerini çekin.")
    st.stop()

# --- Parametre Seçimleri ---
st.sidebar.header("⚙️ Hesaplama Parametreleri")

calc_mode = st.sidebar.radio(
    "🎯 Hesaplama Ölçeği (Önemli):",
    [
        "Seçilen Dönem (Dönemsel)",
        "Yıllıklaştırılmış Projeksiyon (252 Gün)",
    ],
    index=0 if st.session_state.get("calculation_mode", "period") == "period" else 1,
    help="3 aylık veri çektiğinizde 'Seçilen Dönem' seçeneği 252 yerine tam olarak çektiğiniz gün sayısı kadar (T) getiri ve risk hesaplar.",
)
mode_code = "period" if "Dönem" in calc_mode else "annualized"
st.session_state["calculation_mode"] = mode_code

return_type = st.sidebar.radio(
    "Getiri Tipi:",
    ["Basit Getiri", "Log Getiri"],
    index=0,
)

annual_rf_pct = st.sidebar.number_input(
    "Yıllık Risksiz Faiz Oranı (%) [Örn: Politika Faizi]:",
    min_value=0.0,
    max_value=100.0,
    value=float(st.session_state.get("risk_free_rate_pct", 45.0)),
    step=1.0,
)
st.session_state["risk_free_rate_pct"] = annual_rf_pct
annual_rf = annual_rf_pct / 100.0

# Hesaplamaları Motor Üzerinden Yap
returns = calculate_returns(prices, return_type)
market_params = get_market_parameters(
    returns=returns,
    prices=prices,
    mode=mode_code,
    annual_trading_days=252,
    annual_risk_free_rate=annual_rf,
)

actual_days = market_params["actual_days"]
time_factor = market_params["time_factor"]
scaled_rf = market_params["risk_free_rate"]

# Bilgi Kartı
st.info(f"""
💡 **Şu anki mod:** **{calc_mode}**  
* Çekilen Veri: **{actual_days} İşlem Günü**  
* Uygulanan Zaman Çarpanı: **{int(time_factor)}**  
* Kullanılan Risksiz Faiz: **%{scaled_rf * 100:.2f}** *(Yıllık %{annual_rf_pct:.1f} faizin {actual_days} güne indirgenmiş hali)*
""")

st.markdown("---")

# =============================================================================
# 1. HİSSE GETİRİ VE VOLATİLİTE KARŞILAŞTIRMA TABLOSU
# =============================================================================
st.subheader("📈 Hisse Bazlı Getiri ve Risk Tablosu")
st.caption("Aşağıdaki tabloda hisselerin günlük, seçilen dönemsel ve 252 günlük yıllıklandırılmış getiri/riskleri yan yana sunulmuştur.")

daily_mean = returns.mean()
daily_std = returns.std()

comp_rows = []
for col in prices.columns:
    cum_ret = ((prices[col].iloc[-1] / prices[col].iloc[0]) - 1) * 100
    mean_period_ret = daily_mean[col] * actual_days * 100
    ann_ret = daily_mean[col] * 252 * 100
    
    period_vol = daily_std[col] * np.sqrt(actual_days) * 100
    ann_vol = daily_std[col] * np.sqrt(252) * 100

    comp_rows.append({
        "Sembol": col,
        "Şirket": BIST_HISSELER.get(col, col),
        f"Kümülatif Net Getiri (%)": round(cum_ret, 2),
        f"Dönemsel Ortalama Getiri ({actual_days} Gün) (%)": round(mean_period_ret, 2),
        "Yıllıklaştırılmış Getiri (252 Gün) (%)": round(ann_ret, 2),
        f"Dönemsel Volatilite ({actual_days} Gün) (%)": round(period_vol, 2),
        "Yıllık Volatilite (252 Gün) (%)": round(ann_vol, 2),
        "Günlük Volatilite (%)": round(daily_std[col] * 100, 3),
    })

df_comp = pd.DataFrame(comp_rows)
st.dataframe(df_comp, use_container_width=True, hide_index=True)

st.markdown("---")

# =============================================================================
# 2. KOVARYANS VE KORELASYON MATRİSİ
# =============================================================================
st.subheader("🔢 Risk Matrisi (Kovaryans & Korelasyon)")

mat_choice = st.radio(
    "Görüntülenecek Matris:",
    [
        f"Dönemsel Kovaryans Matrisi ({actual_days} İşlem Günü)",
        "Yıllık Kovaryans Matrisi (252 Gün Projeksiyonu)",
        "Günlük Kovaryans Matrisi (1 Günlük)",
        "Korelasyon Matrisi (Değişmez)",
    ],
    horizontal=True,
)

if "Dönemsel" in mat_choice:
    cur_matrix = returns.cov() * actual_days
    cap = f"📌 Bu matris, çektiğiniz **{actual_days} işlem günü** boyunca hisselerin birlikte hareket riskini gösterir (Günlük Cov × {actual_days})."
elif "Yıllık" in mat_choice:
    cur_matrix = returns.cov() * 252
    cap = "📌 Bu matris, hisselerin günlük kovaryansının 252 ile çarpılmasıyla elde edilen geleneksel yıllık kovaryans matrisidir."
elif "Günlük" in mat_choice:
    cur_matrix = returns.cov()
    cap = "📌 Günlük kapanış getirileri üzerinden doğrudan hesaplanan 1 günlük varyans/kovaryans değerleri."
else:
    cur_matrix = returns.corr()
    cap = "📌 Hisseler arasındaki doğrusal ilişki derecesi (-1 ile +1 arası). Zaman ufkundan (günlük, dönemsel, yıllık) bağımsızdır."

st.caption(cap)

c_tab, c_heat = st.columns([1, 1])

with c_tab:
    st.markdown("##### 📋 Matris Verileri")
    st.dataframe(cur_matrix.round(6), use_container_width=True)

with c_heat:
    st.markdown("##### 🎨 Isı Haritası (Heatmap)")
    fig_heat = px.imshow(
        cur_matrix.round(4),
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis" if "Korelasyon" not in mat_choice else "RdBu_r",
        template="plotly_dark",
    )
    fig_heat.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_heat, use_container_width=True)

# İndirme Butonu
csv_data = cur_matrix.to_csv().encode("utf-8-sig")
st.download_button(
    f"⬇️ {mat_choice} (CSV İndir)",
    data=csv_data,
    file_name=f"kovaryans_matrisi_{mode_code}.csv",
    mime="text/csv",
)

st.markdown("---")

# =============================================================================
# 3. EĞİTİCİ AÇIKLAMA (SİZİN SORDUĞUNUZ SORUNUN YANITI)
# =============================================================================
with st.expander("📚 Neden Dönemsel Kovaryans? (Yıllık vs. Dönemsel Farkı Nedir?)", expanded=True):
    st.markdown(f"""
    #### 🔍 Sık Yapılan Hata: Zorunlu 252 Çarpanı
    Geleneksel yazılımlar, siz 3 ay (yaklaşık **{actual_days} gün**) veri çekseniz bile getiri ve kovaryansı otomatik olarak **252** ile çarpar.
    
    * **Sonuç Ne Olur?** Eğer seçtiğiniz o 3 ayda borsa çok iyi geçmiş ve bir hisse %25 kazandırmışsa, sistem bunu 252 güne yayarak ekranda **%100+ beklenen getiri** yazar. Bu ise geleceğe yönelik gerçekçi olmayan bir yanılsama üretir.
    * **Dönemsel Mod Ne Yapar?** "Seçilen Dönem" modu aktif olduğunda, kovaryans ve getiri doğrudan **{actual_days} gün** ile çarpılır. Böylece ekranda **"Bu 3 ayda beklenen getiri: %{daily_mean.mean() * actual_days * 100:.1f}"** gibi gerçekçi ve ayağı yere basan sayılar görürsünüz.
    * **Korelasyon Etkilenir mi?** Hayır! Zaman çarpanı hem payı hem paydayı eşit oranda etkilediği için **korelasyon matrisi her iki modda da birebir aynıdır**.
    """)

st.info("👉 Şimdi sol menüden **🧠 3_Markowitz_Optimizasyon** sayfasına giderek seçtiğiniz bu döneme göre en optimal portföy dağılımlarını hesaplayabilirsiniz.")

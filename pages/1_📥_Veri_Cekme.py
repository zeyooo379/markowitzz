"""
1_📥_Veri_Cekme: BIST hisse senedi fiyat verilerini çekme ve önizleme
"""

from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.constants import BIST_HISSELER, POPULER_PORTFOYLER, RENKLER
from utils.stock_data import hisse_verisi_cek

st.set_page_config(page_title="Veri Çekme — Markowitzz", page_icon="📥", layout="wide")

st.title("📥 BIST Veri Çekme & Zaman Penceresi")
st.markdown("""
Analiz etmek istediğiniz hisseleri ve zaman periyodunu seçin. 
İster **3 aylık**, ister **yıllık** veri çekebilirsiniz; sistem hesaplamaları seçtiğiniz bu döneme göre ölçeklendirecektir.
""")

# --- Yan Menü / Kontrol Alanı ---
st.sidebar.header("⚙️ Veri Seçenekleri")

# Hazır Portföy Seçimi
hazir_sepet = st.sidebar.selectbox(
    "Hazır Sepet Yükle (İsteğe Bağlı):",
    options=["Özel Seçim"] + list(POPULER_PORTFOYLER.keys()),
)

if hazir_sepet != "Özel Seçim":
    varsayilan_hisseler = POPULER_PORTFOYLER[hazir_sepet]
else:
    varsayilan_hisseler = st.session_state.get(
        "selected_symbols", ["THYAO", "GARAN", "KCHOL", "TUPRS", "BIMAS"]
    )

secilenler = st.sidebar.multiselect(
    "Hisse Senetleri:",
    options=list(BIST_HISSELER.keys()),
    default=varsayilan_hisseler,
    format_func=lambda x: f"{x} — {BIST_HISSELER.get(x, x)}",
    help="En az 2 hisse seçmelisiniz.",
)

# Zaman Aralığı Hızlı Butonları
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Zaman Aralığı")

periyot_tipi = st.sidebar.radio(
    "Periyot Seçimi:",
    ["3 Ay (Çeyreklik)", "1 Ay", "6 Ay", "1 Yıl", "Yılbaşından Bugüne (YBB)", "Özel Tarih Aralığı"],
    index=0,
)

bugun = datetime.today()

if periyot_tipi == "1 Ay":
    baslangic = bugun - timedelta(days=30)
    bitis = bugun
elif periyot_tipi == "3 Ay (Çeyreklik)":
    baslangic = bugun - timedelta(days=90)
    bitis = bugun
elif periyot_tipi == "6 Ay":
    baslangic = bugun - timedelta(days=180)
    bitis = bugun
elif periyot_tipi == "1 Yıl":
    baslangic = bugun - timedelta(days=365)
    bitis = bugun
elif periyot_tipi == "Yılbaşından Bugüne (YBB)":
    baslangic = datetime(bugun.year, 1, 1)
    bitis = bugun
else:
    baslangic = st.sidebar.date_input("Başlangıç Tarihi:", bugun - timedelta(days=90))
    bitis = st.sidebar.date_input("Bitiş Tarihi:", bugun)

str_start = baslangic.strftime("%Y-%m-%d")
str_end = bitis.strftime("%Y-%m-%d")

veri_cek_btn = st.sidebar.button("🚀 Verileri Çek ve Güncelle", type="primary", use_container_width=True)

# Otomatik veya Butonla Veri Çekme
if veri_cek_btn or st.session_state.get("prices") is None:
    if len(secilenler) < 2:
        st.warning("⚠️ Lütfen portföy analizi için en az 2 hisse senedi seçin.")
    else:
        with st.spinner("BIST verileri yfinance üzerinden çekiliyor..."):
            df_prices = hisse_verisi_cek(secilenler, str_start, str_end)
            if not df_prices.empty and len(df_prices) >= 10:
                st.session_state["prices"] = df_prices
                st.session_state["selected_symbols"] = list(df_prices.columns)
                st.session_state["start_date"] = str_start
                st.session_state["end_date"] = str_end
                st.session_state["period_name"] = periyot_tipi
                st.success(f"✅ {len(df_prices.columns)} hisse için toplam **{len(df_prices)} işlem günü** verisi başarıyla yüklendi!")
            else:
                st.error("❌ Veri çekilemedi veya seçilen tarih aralığında yeterli işlem günü (en az 10) bulunamadı.")

# --- Ekran Görünümü ---
prices = st.session_state.get("prices")

if prices is not None and not prices.empty:
    n_days = len(prices)
    start_fmt = prices.index[0].strftime("%d.%m.%Y")
    end_fmt = prices.index[-1].strftime("%d.%m.%Y")
    
    # Bilgi Metrikleri
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("İşlem Günü Sayısı (T)", f"{n_days} Gün", help="Seçtiğiniz dönemdeki gerçek borsa seans sayısı")
    m2.metric("Takvim Dönemi", st.session_state.get("period_name", periyot_tipi))
    m3.metric("Tarih Aralığı", f"{start_fmt} - {end_fmt}")
    m4.metric("Seçilen Hisse Sayısı", len(prices.columns))

    st.markdown("---")
    
    # Normalize Edilmiş Getiri Karşılaştırma Grafiği (Başlangıç = 100)
    st.subheader("📈 Normalize Edilmiş Performans Grafiği (Baz = 100)")
    st.caption("Seçilen dönemin ilk günündeki fiyat 100 kabul edilerek hisselerin kümülatif seyri gösterilmektedir.")
    
    norm_prices = (prices / prices.iloc[0]) * 100
    fig = px.line(
        norm_prices,
        x=norm_prices.index,
        y=norm_prices.columns,
        labels={"value": "Kümülatif Değer (Baz 100)", "Date": "Tarih", "variable": "Hisse"},
        template="plotly_dark",
    )
    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Son Fiyatlar Tablosu
    st.subheader("📋 Çekilen Veri Özeti")
    summary_data = []
    for col in prices.columns:
        ilk = prices[col].iloc[0]
        son = prices[col].iloc[-1]
        degisim_pct = ((son / ilk) - 1) * 100
        summary_data.append({
            "Sembol": col,
            "Şirket": BIST_HISSELER.get(col, col),
            "Başlangıç Fiyatı (TL)": round(ilk, 2),
            "Son Kapanış (TL)": round(son, 2),
            f"Seçilen Dönem Net Getiri (%)": round(degisim_pct, 2),
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    st.info("👉 Şimdi sol menüden **📊 2_Donemsel_Risk_Getiri** sayfasına geçerek dönemsel kovaryans ve getiri dağılımlarını inceleyebilirsiniz.")
else:
    st.info("👆 Lütfen sol menüdeki **🚀 Verileri Çek ve Güncelle** butonuna tıklayarak veriyi yükleyin.")

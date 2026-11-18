"""
Markowitzz — Dönemsel ve Dinamik BIST Portföy Optimizasyonu
Ana Giriş Sayfası & Çekilen Hisselerin Dönemsel Ortalama Getirileri
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.constants import BIST_HISSELER, RENKLER
from utils.stock_data import hisse_verisi_cek

st.set_page_config(
    page_title="Markowitzz 📈 BIST Portföy Analizi",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session State Hazırlığı
if "selected_symbols" not in st.session_state:
    st.session_state["selected_symbols"] = ["THYAO", "GARAN", "KCHOL", "TUPRS", "BIMAS"]
if "return_type" not in st.session_state:
    st.session_state["return_type"] = "Basit Getiri"
if "risk_free_rate_pct" not in st.session_state:
    st.session_state["risk_free_rate_pct"] = 45.0
if "calculation_mode" not in st.session_state:
    st.session_state["calculation_mode"] = "period"

# Varsayılan veri yükleme (Eğer hafızada yoksa otomatik 3 aylık veri yükle)
if "prices" not in st.session_state or st.session_state["prices"] is None:
    today = datetime.today()
    start_dt = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    end_dt = today.strftime("%Y-%m-%d")
    with st.spinner("Varsayılan 3 aylık BIST hisse verileri yükleniyor..."):
        df_init = hisse_verisi_cek(st.session_state["selected_symbols"], start_dt, end_dt)
        if not df_init.empty:
            st.session_state["prices"] = df_init
            st.session_state["start_date"] = start_dt
            st.session_state["end_date"] = end_dt
            st.session_state["period_name"] = "3 Ay (Çeyreklik)"

# Özel CSS Tasarımı
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.15) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 26px 30px;
        margin-bottom: 25px;
    }
    
    .returns-panel {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(30, 41, 59, 0.95) 100%);
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 25px;
    }
    
    .feature-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        height: 100%;
        transition: transform 0.2s, border-color 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        border-color: #38BDF8;
    }
    
    .badge-period {
        background-color: #10B981;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-annual {
        background-color: #3B82F6;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Hero Başlık Alanı
st.markdown("""
<div class="hero-container">
    <h1 style="color: #F8FAFC; margin-bottom: 8px;">📊 Markowitzz: BIST Dinamik & Dönemsel Portföy Optimizasyonu</h1>
    <p style="color: #94A3B8; font-size: 1.15rem; margin-bottom: 12px;">
        Geleneksel 252 günlük yıllıklandırma zorunluluğuna son! Çektiğiniz veri ister <b>1 ay</b>, ister <b>3 ay</b>, ister <b>1 yıl</b> olsun; 
        hisselerin getiri ve riskini <b>tam olarak o dönemin gerçek işlem günlerine göre</b> hesaplayın.
    </p>
    <div>
        <span class="badge-period">✨ Dönemsel Hesaplama Desteği</span>
        <span class="badge-annual">📅 Yıllıklaştırılmış Projeksiyon</span>
        <span class="badge-period">🎯 Gerçekçi Sharpe & Kovaryans</span>
    </div>
</div>
""", unsafe_allow_html=True)

prices = st.session_state.get("prices")

# =============================================================================
# ⭐ ÇEKİLEN HİSSELERİN O DÖNEMDEKİ ORTALAMA GETİRİLERİ (İSTENEN ÖZEL BÖLÜM)
# =============================================================================
if prices is not None and not prices.empty:
    daily_returns = prices.pct_change(fill_method=None).dropna()
    actual_days = len(daily_returns)
    start_fmt = prices.index[0].strftime("%d.%m.%Y")
    end_fmt = prices.index[-1].strftime("%d.%m.%Y")
    period_name = st.session_state.get("period_name", f"{actual_days} İşlem Günü")

    # Günlük ortalama ve dönemsel ortalama getiriler
    daily_mean_pct = daily_returns.mean() * 100
    period_mean_pct = daily_mean_pct * actual_days
    ann_mean_pct = daily_mean_pct * 252
    
    # Kümülatif net getiri (başlangıçtan bitişe)
    cum_returns_pct = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100

    # Portföy sepet ortalamaları
    basket_avg_period_return = float(period_mean_pct.mean())
    basket_avg_daily_return = float(daily_mean_pct.mean())
    best_stock = period_mean_pct.idxmax()
    worst_stock = period_mean_pct.idxmin()

    st.markdown('<div class="returns-panel">', unsafe_allow_html=True)
    st.markdown(f"""
    <h3 style="color: #10B981; margin-bottom: 4px;">📈 Çekilen Hisselerin Seçilen Dönemdeki Ortalama Getirileri</h3>
    <p style="color: #CBD5E1; font-size: 0.95rem; margin-bottom: 15px;">
        Zaman Penceresi: <b>{period_name}</b> ({start_fmt} - {end_fmt}) | Toplam: <b>{actual_days} Borsa İşlem Günü</b>
    </p>
    """, unsafe_allow_html=True)

    # Özet Kartlar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Sepet Ortalama Dönem Getirisi",
        f"%{basket_avg_period_return:.2f}",
        delta=f"Günlük Ort: %{basket_avg_daily_return:.3f}",
    )
    m2.metric(
        f"🏆 En Çok Kazandıran ({period_name})",
        f"{best_stock}",
        delta=f"%{period_mean_pct[best_stock]:.2f}",
    )
    m3.metric(
        f"📉 En Az Kazandıran / Düşen",
        f"{worst_stock}",
        delta=f"%{period_mean_pct[worst_stock]:.2f}",
        delta_color="inverse",
    )
    m4.metric(
        "Aktif İncelenen Hisse",
        f"{len(prices.columns)} Hisse",
        delta=f"{actual_days} Günlük Veri",
    )

    st.markdown("---")

    # Çubuk Grafik & Tablo
    g_col, t_col = st.columns([3, 2])

    df_returns_summary = pd.DataFrame({
        "Sembol": prices.columns,
        "Şirket": [BIST_HISSELER.get(s, s) for s in prices.columns],
        f"Dönemsel Ortalama Getiri ({actual_days} Gün) (%)": period_mean_pct.round(2).values,
        "Dönem Kümülatif Net Getirisi (%)": cum_returns_pct.round(2).values,
        "Ortalama Günlük Getiri (%)": daily_mean_pct.round(3).values,
        "Yıllıklaştırılmış Eşdeğeri (252 Gün) (%)": ann_mean_pct.round(2).values,
    }).sort_values(by=f"Dönemsel Ortalama Getiri ({actual_days} Gün) (%)", ascending=False)

    with g_col:
        fig_bar = px.bar(
            df_returns_summary,
            x="Sembol",
            y=f"Dönemsel Ortalama Getiri ({actual_days} Gün) (%)",
            color=f"Dönemsel Ortalama Getiri ({actual_days} Gün) (%)",
            color_continuous_scale="Viridis",
            text=f"Dönemsel Ortalama Getiri ({actual_days} Gün) (%)",
            template="plotly_dark",
            title=f"Hisselerin {period_name} ({actual_days} İşlem Günü) Boyunca Ortalama Getirisi (%)",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    with t_col:
        st.markdown(f"##### 📋 Hisse Getiri Tablosu")
        st.dataframe(
            df_returns_summary[["Sembol", f"Dönemsel Ortalama Getiri ({actual_days} Gün) (%)", "Dönem Kümülatif Net Getirisi (%)", "Ortalama Günlük Getiri (%)"]],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Bilgi ve Kılavuz Kartları
st.subheader("🧭 Sayfa Kılavuzu & Yetenekler")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #38BDF8;">📥 1. Esnek Veri Çekme</h3>
        <p style="color: #CBD5E1; font-size: 0.95rem;">
            1 Ay, 3 Ay, 6 Ay, 1 Yıl veya özel tarih aralığıyla BIST hisselerinin fiyatlarını anında çekebilir, hazır sektör sepetlerini yükleyebilirsiniz.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #10B981;">📊 2. Dönemsel Risk & Kovaryans</h3>
        <p style="color: #CBD5E1; font-size: 0.95rem;">
            Seçtiğiniz dönemin gerçek kovaryans ve korelasyon matrislerini inceleyebilir, yıllıklandırma ile dönemsel farkları kıyaslayabilirsiniz.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #F59E0B;">🧠 3. Markowitz & 10 Portföy</h3>
        <p style="color: #CBD5E1; font-size: 0.95rem;">
            Etkin Sınır üzerinde 5 tanesi Sharpe altı, 5 tanesi Sharpe üstü olmak üzere 10 stratejik portföyü ve kendi özel sepetinizi test edebilirsiniz.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.info("👉 Farklı hisseler veya farklı bir zaman aralığı (örn: 1 Ay, 6 Ay) seçmek için sol menüden **📥 1_Veri_Cekme** sayfasına geçebilirsiniz.")

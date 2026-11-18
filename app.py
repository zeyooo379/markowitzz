"""
Markowitzz — Dönemsel ve Dinamik BIST Portföy Optimizasyonu
Ana Giriş Sayfası
"""

import streamlit as st
from utils.constants import RENKLER

st.set_page_config(
    page_title="Markowitzz 📈 BIST Portföy Analizi",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session State Hazırlığı
if "prices" not in st.session_state:
    st.session_state["prices"] = None
if "selected_symbols" not in st.session_state:
    st.session_state["selected_symbols"] = ["THYAO", "GARAN", "KCHOL", "TUPRS", "BIMAS"]
if "return_type" not in st.session_state:
    st.session_state["return_type"] = "Basit Getiri"
if "risk_free_rate_pct" not in st.session_state:
    st.session_state["risk_free_rate_pct"] = 45.0
if "calculation_mode" not in st.session_state:
    st.session_state["calculation_mode"] = "period"  # 'period' veya 'annualized'

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
        padding: 30px;
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
        Geleneksel 252 günlük yıllıklandırma kısıtlamasından kurtulun! Çektiğiniz veri ister <b>1 ay</b>, ister <b>3 ay</b>, ister <b>2 yıl</b> olsun; 
        getiri, kovaryans ve risk analizini <b>tam olarak seçtiğiniz döneme göre</b> hesaplayın.
    </p>
    <div>
        <span class="badge-period">✨ Dönemsel Hesaplama Desteği</span>
        <span class="badge-annual">📅 Yıllıklaştırılmış Projeksiyon</span>
        <span class="badge-period">🎯 Gerçekçi Sharpe & Kovaryans</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Bilgi ve Kılavuz Kartları
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #38BDF8;">📥 1. Esnek Veri Çekme</h3>
        <p style="color: #CBD5E1; font-size: 0.95rem;">
            1 Ay, 3 Ay, 6 Ay, 1 Yıl gibi hazır periyotlar veya istediğiniz özel tarih aralığıyla BIST hisselerinin fiyatlarını anında çekin.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #10B981;">📊 2. Dönemsel Risk & Kovaryans</h3>
        <p style="color: #CBD5E1; font-size: 0.95rem;">
            3 aylık veride 252 katsayısı kullanıp afaki getiriler görmek yerine, tam olarak o 3 ayın gerçekleşen kovaryans matrisini ve getirisini inceleyin.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #F59E0B;">🧠 3. Markowitz & Etkin Sınır</h3>
        <p style="color: #CBD5E1; font-size: 0.95rem;">
            Minimum Volatilite, Maksimum Sharpe ve Etkin Sınır (Efficient Frontier) analizlerini dönemsel veya yıllık ölçekte tek tıkla çalıştırın.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Hızlı Başlangıç
st.subheader("🚀 Hızlı Başlangıç")
st.info("""
Analize başlamak için sol menüden **📥 1_Veri_Cekme** sayfasına gidin, hisselerinizi ve analiz etmek istediğiniz zaman aralığını (örn: 3 Ay) seçin!
""")

# Durum Bilgisi
if st.session_state["prices"] is not None and not st.session_state["prices"].empty:
    p = st.session_state["prices"]
    st.success(f"✅ Hafızada yüklü veri mevcut: **{len(p.columns)} Hisse**, **{len(p)} İşlem Günü** ({p.index[0].strftime('%d.%m.%Y')} - {p.index[-1].strftime('%d.%m.%Y')})")
else:
    st.caption("ℹ️ Henüz veri çekilmedi. Sol menüden '1_Veri_Cekme' sayfasına geçerek verinizi yükleyebilirsiniz.")

# Markowitzz 📈
### BIST Dinamik & Dönemsel Portföy Optimizasyonu (Markowitz Modern Portfolio Theory)

Borsa İstanbul (BIST) hisse senetleri için geliştirilmiş, **geleneksel 252 günlük yıllıklaştırma zorunluluğunu esneten**, kullanıcının seçtiği periyot (1 ay, 3 ay, 6 ay vb.) üzerinden doğrudan dönemsel getiri ve kovaryans hesabı yapabilen modern bir portföy analizi ve optimizasyon platformudur.

---

## 🎯 Temel Fark: Neden Markowitzz?

Geleneksel portföy yazılımları, kullanıcı **3 aylık (yaklaşık 63 işlem günü)** veri çektiğinde dahi günlük ortalama getiriyi ve kovaryansı otomatik olarak **252** ile çarpar. Bu durum, 3 aylık kısa vadeli bir rallide yıllık %100+ afaki ve yanıltıcı getiriler türetir.

**Markowitzz ile:**
- **Dönemsel Ölçek:** 3 aylık veri çekildiğinde getiri ve risk tam olarak o 3 aylık dönemin ($T$ gün) kümülatif büyüklükleri üzerinden hesaplanır.
- **Yıllıklaştırılmış Ölçek:** İstenirse tek tıkla standart 252 günlük projeksiyona geçiş yapılır.
- **Risksiz Faiz İndirgemesi:** Yıllık politika faizi (%45 vb.), seçilen döneme ($T/252$) bileşik olarak indirgenir; böylece Sharpe oranı gerçeğe tam uygun hesaplanır.

---

## 🚀 Özellikler

1. **📥 Veri Çekme & Zaman Penceresi:**
   - 1 Ay, 3 Ay (Çeyreklik), 6 Ay, 1 Yıl veya Özel Tarih Aralığı.
   - Hazır BIST sepetleri (Bankacılık, Sanayi, Havacılık, Teknoloji vb.).
   - Normalize edilmiş (Baz 100) karşılaştırmalı performans grafikleri.
2. **📊 Dönemsel Risk & Kovaryans Matrisi:**
   - Dönemsel Kovaryans vs. Yıllık Kovaryans vs. Günlük Kovaryans.
   - İnteraktif ısı haritası (Heatmap) ve CSV dışa aktarma.
   - Hisseler arası korelasyon matrisi.
3. **🧠 Markowitz Optimizasyonu & Etkin Sınır:**
   - Maksimum Sharpe Portföyü (Teğet Portföy).
   - Minimum Volatilite Portföyü.
   - Eşit Ağırlıklı Portföy (1/N benchmark).
   - Etkin Sınır (Efficient Frontier) interaktif eğrisi ve hisse dağılım pasta grafikleri.
4. **📈 Monte Carlo Simülasyonu & Sermaye Pazarı Doğrusu (CML):**
   - 2000+ rastgele portföy simülasyonu.
   - Sermaye Pazarı Doğrusu (CML) teğet çizgisi ve risksiz varlık bileşimi.

---

## 📦 Kurulum ve Çalıştırma

```bash
# 1. Proje dizinine geçin
cd markowitzz

# 2. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 3. Streamlit uygulamasını başlatın
streamlit run app.py
```

---

## 🛠️ Teknolojiler
- **Streamlit** — Web arayüzü
- **yfinance** — Canlı ve geçmiş BIST verileri
- **Pandas & NumPy** — Finansal veri işleme ve matris matematiği
- **Scipy (optimize.minimize / SLSQP)** — Kısıtlı portföy optimizasyonu
- **Plotly** — Etkileşimli finans grafikleri

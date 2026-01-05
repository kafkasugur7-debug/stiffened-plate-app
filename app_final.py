import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------------
# AYARLAR VE TASARIM
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Stiffened Plate Frekans Tahmincisi",
    page_icon="🏗️",
    layout="wide"
)

st.markdown("""
# 🏗️ Stiffened Plate Boyutsuz Frekans Tahmin Modülü
Bu arayüz, **Number of Stiffeners, Plate Slenderness Ratio, Stiffener Height to Plate Thickness Ratio, Stiffener Width to Plate Thickness Ratio** parametrelerini kullanarak ML modelleri ve **GEP formülü** ile tahmin yapar.
""")

# ----------------------------------------------------------------
# 1. DOSYA YOLU VE MODEL YÜKLEME
# ----------------------------------------------------------------
MODEL_KLASORU = "."

@st.cache_resource
def modelleri_yukle():
    yuklenen_modeller = {}
    model_listesi = {
        'ANN_(MLP).pkl': 'Yapay Sinir Ağı (ANN)',
        'Gradient_Boosting.pkl': 'Gradient Boosting',
        'Linear_Regression.pkl': 'Lineer Regresyon',
        'Random_Forest.pkl': 'Random Forest',
        'Ridge_Regression.pkl': 'Ridge Regresyon'
    }
    
    for dosya_adi, gorunen_isim in model_listesi.items():
        tam_yol = os.path.join(MODEL_KLASORU, dosya_adi)
        try:
            model = joblib.load(tam_yol)
            yuklenen_modeller[gorunen_isim] = model
        except:
            pass
            
    return yuklenen_modeller

ml_models = modelleri_yukle()

# ----------------------------------------------------------------
# 2. GEP FORMÜLÜ (v4)
# ----------------------------------------------------------------
def gep_hesapla_v4(nos, bpt, shpt, swpt):
    # GeneXproTools v5.0 Model (v4)
    G1C0 = 2.41533020894101
    G2C0 = -4.22895804681579
    G2C2 = 5.05838626177623
    G3C4 = 5.49221888709335
    G3C9 = 0.230950738560598

    d0, d1, d2, d3 = nos, bpt, shpt, swpt

    # Terim 1
    term1 = ((d3 * d1)**0.25) * (((G1C0 + d2) - d3) - (d2 / (d0 + 1e-10)))
    # Terim 2
    term2 = (d3 - (np.abs(d3) * (d0 - G2C2))) + G2C0
    # Terim 3
    term3 = ((np.maximum(0, d0 - G3C9))**0.25) * (G3C4 * (d3 + d2))

    return term1 + term2 + term3

# ----------------------------------------------------------------
# 3. KULLANICI ARAYÜZÜ (INPUT)
# ----------------------------------------------------------------
st.sidebar.header("⚙️ Parametre Girişi")

# SİZİN İSTEDİĞİNİZ DEĞERLER VE FORMATLAR:
nos_val  = st.sidebar.number_input("nos (Number of Stiffeners)", value=2.0, step=1.0, format="%.4f")
bpt_val  = st.sidebar.number_input("bpt (Breadth/Thickness)", value=140.0, step=0.5, format="%.4f")
shpt_val = st.sidebar.number_input("shpt (Shape Parameter)", value=9.0, step=0.1, format="%.4f")
swpt_val = st.sidebar.number_input("swpt (Stiffener Width)", value=1.5, step=0.1, format="%.4f")

# --- KRİTİK DÜZELTME: Session State Kullanımı ---
if 'hesaplandi' not in st.session_state:
    st.session_state['hesaplandi'] = False

# Butona basılınca state'i True yapıyoruz
if st.sidebar.button("🚀 Hesapla ve Analiz Et", type="primary"):
    st.session_state['hesaplandi'] = True

# ----------------------------------------------------------------
# 4. HESAPLAMA VE GÖRSELLEŞTİRME
# ----------------------------------------------------------------
# Artık buton yerine session_state kontrol ediyoruz
if st.session_state['hesaplandi']:
    st.divider()
    
    # --- A) TEKİL TAHMİN ---
    input_data = {'nos': nos_val, 'bpt': bpt_val, 'shpt': shpt_val, 'swpt': swpt_val}
    input_df = pd.DataFrame([input_data])
    
    sonuclar = []
    
    # ML Modelleri
    if ml_models:
        for isim, model in ml_models.items():
            try:
                tahmin = model.predict(input_df)[0]
                if isinstance(tahmin, (np.ndarray, list)): 
                    tahmin = tahmin.item()
                sonuclar.append({"Model": isim, "Tahmin": tahmin, "Tür": "ML"})
            except: 
                sonuclar.append({"Model": isim, "Tahmin": 0.0, "Tür": "HATA"})

    # GEP Modeli
    try:
        gep_sonuc = gep_hesapla_v4(nos_val, bpt_val, shpt_val, swpt_val)
        sonuclar.append({"Model": "GEP Formülü (v4)", "Tahmin": gep_sonuc, "Tür": "Matematiksel"})
    except:
        sonuclar.append({"Model": "GEP Formülü (v4)", "Tahmin": 0.0, "Tür": "HATA"})
    
    df_sonuc = pd.DataFrame(sonuclar)
    
    # Format hatasını önleyen düzeltme
    df_sonuc["Tahmin"] = pd.to_numeric(df_sonuc["Tahmin"], errors='coerce').fillna(0.0)
    df_sonuc = df_sonuc.sort_values(by="Tahmin", ascending=False)

    # Sekmeler
    tab1, tab2 = st.tabs(["📊 Tahmin Özeti", "📈 Duyarlılık (Trend) Analizi"])

    # --- TAB 1: STANDART SONUÇLAR ---
    with tab1:
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.subheader("📋 Sonuç Tablosu")
            st.dataframe(
                df_sonuc.style.background_gradient(cmap="Oranges", subset=["Tahmin"])
                              .format({"Tahmin": "{:.4f}"}), 
                use_container_width=True
            )
            
            gecerli_tahminler = df_sonuc[df_sonuc['Tahmin'] > 0.0001]['Tahmin']
            if not gecerli_tahminler.empty:
                st.info(f"Ortalama Frekans: {gecerli_tahminler.mean():.4f}")

        with col2:
            st.subheader("📊 Tahmin Karşılaştırması")
            fig, ax = plt.subplots(figsize=(8, 4))
            renkler = ['#e74c3c' if 'GEP' in m else '#3498db' for m in df_sonuc['Model']]
            bars = ax.barh(df_sonuc['Model'], df_sonuc['Tahmin'], color=renkler)
            ax.set_xlabel('Boyutsuz Frekans')
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            for bar in bars:
                width = bar.get_width()
                if width > 0:
                    ax.text(width*1.01, bar.get_y()+bar.get_height()/2, f'{width:.2f}', va='center', fontsize=9)
            st.pyplot(fig)

    # --- TAB 2: DUYARLILIK ANALİZİ ---
    with tab2:
        st.subheader("📈 Parametrik Trend Analizi")
        st.markdown("Seçilen değişkenin değeri **±%50** değiştirildiğinde modellerin tepkisi:")
        
        # Selectbox artık sayfayı yenilese bile 'hesaplandi' True olduğu için buraya tekrar girecek
        degisken = st.selectbox("Analiz edilecek değişkeni seçin:", ['nos', 'bpt', 'shpt', 'swpt'])
        
        base_val = input_data[degisken]
        if base_val == 0: base_val = 1 
        
        x_axis = np.linspace(base_val * 0.5, base_val * 1.5, 50) 
        
        plt.figure(figsize=(10, 6))
        
        # ML Modelleri Çizimi
        for isim, model in ml_models.items():
            temp_input = input_data.copy()
            temp_df = pd.DataFrame([temp_input] * 50)
            temp_df[degisken] = x_axis
            
            try:
                preds = model.predict(temp_df)
                plt.plot(x_axis, preds, label=isim, alpha=0.6, linewidth=2)
            except: pass
            
        # GEP Modeli Çizimi
        y_gep = []
        for x in x_axis:
            vals = input_data.copy()
            vals[degisken] = x
            try:
                val = gep_hesapla_v4(vals['nos'], vals['bpt'], vals['shpt'], vals['swpt'])
                y_gep.append(val)
            except:
                y_gep.append(0)
        
        plt.plot(x_axis, y_gep, label="GEP Formülü", color="red", linewidth=3, linestyle="--")
        plt.axvline(x=base_val, color='black', linestyle=':', label="Seçili Değer")
        
        plt.title(f"{degisken} Değişkenine Göre Frekans Değişimi")
        plt.xlabel(degisken)
        plt.ylabel("Boyutsuz Frekans")
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(plt)

else:
    st.info("👈 Sonuçları görmek için sol menüden değerleri girip butona basın.")
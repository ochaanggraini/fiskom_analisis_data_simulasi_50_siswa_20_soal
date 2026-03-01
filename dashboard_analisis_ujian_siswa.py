
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Dashboard Analisis Hasil Ujian", layout="wide")
st.title("📊 Dashboard Analisis Hasil Ujian Siswa")
st.markdown("Analisis berbasis data 50 siswa dan 20 soal")

# ==========================================================
# LOAD DATA
# ==========================================================
uploaded_file = st.file_uploader("Upload file Excel data siswa", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    # Hitung total skor
    df["Total_Skor"] = df.sum(axis=1)
    
    # ==========================================================
    # 1️⃣ KPI UTAMA
    # ==========================================================
    mean_total = df["Total_Skor"].mean()
    max_score = df.drop(columns=["Total_Skor"]).shape[1] * 4
    persentase = (mean_total / max_score) * 100
    
    def kategori_nilai(x):
        if x >= 85: return "Sangat Baik"
        elif x >= 70: return "Baik"
        elif x >= 55: return "Cukup"
        else: return "Kurang"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Rata-rata Nilai", f"{mean_total:.2f}")
    col2.metric("🏷️ Kategori", kategori_nilai(persentase))
    col3.metric("👥 Jumlah Siswa", len(df))
    
    st.divider()
    
    # ==========================================================
    # 2️⃣ ANALISIS KESUKARAN SOAL (GAP)
    # ==========================================================
    st.header("2️⃣ Analisis Tingkat Kesukaran Soal")
    
    mean_per_soal = df.drop(columns=["Total_Skor"]).mean()
    gap = 4 - mean_per_soal
    prioritas = gap.idxmax()
    
    fig_gap, ax_gap = plt.subplots(figsize=(8,4))
    ax_gap.bar(gap.index, gap.values)
    ax_gap.set_ylabel("Nilai GAP")
    ax_gap.set_title("GAP Kesukaran per Soal")
    ax_gap.tick_params(axis='x', rotation=90)
    st.pyplot(fig_gap)
    
    st.info(f"📌 Soal paling sulit (prioritas evaluasi): **{prioritas}**")
    
    st.divider()
    
    # ==========================================================
    # 3️⃣ KORELASI ANTAR SOAL
    # ==========================================================
    st.header("3️⃣ Korelasi Antar Soal")
    
    corr = df.drop(columns=["Total_Skor"]).corr()
    
    fig_corr, ax_corr = plt.subplots(figsize=(6,5))
    im = ax_corr.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax_corr)
    
    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_yticks(range(len(corr.columns)))
    ax_corr.set_xticklabels(corr.columns, rotation=90)
    ax_corr.set_yticklabels(corr.columns)
    
    st.pyplot(fig_corr)
    
    st.divider()
    
    # ==========================================================
    # 4️⃣ ANALISIS REGRESI
    # ==========================================================
    st.header("4️⃣ Analisis Regresi (Prediksi Total Skor)")
    
    X = sm.add_constant(df.drop(columns=["Total_Skor"]).iloc[:, :-1])
    y = df["Total_Skor"]
    
    model = sm.OLS(y, X).fit()
    coef = model.params[1:]
    
    fig_reg, ax_reg = plt.subplots(figsize=(8,4))
    ax_reg.bar(coef.index, coef.values)
    ax_reg.axhline(0, linestyle="--")
    ax_reg.set_title("Koefisien Regresi terhadap Total Skor")
    ax_reg.tick_params(axis='x', rotation=90)
    st.pyplot(fig_reg)
    
    st.info(f"📈 Nilai R²: {model.rsquared:.2f}")
    
    st.divider()
    
    # ==========================================================
    # 5️⃣ SEGMENTASI SISWA
    # ==========================================================
    st.header("5️⃣ Segmentasi Siswa (Clustering)")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(columns=["Total_Skor"]))
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster = kmeans.fit_predict(X_scaled)
    
    df["Cluster"] = cluster
    
    cluster_mean = df.groupby("Cluster")["Total_Skor"].mean().sort_values(ascending=False)
    st.subheader("Rata-rata Total Skor per Cluster")
    st.dataframe(cluster_mean)
    
    fig_cluster, ax_cluster = plt.subplots()
    ax_cluster.hist(df["Total_Skor"], bins=10)
    ax_cluster.set_title("Distribusi Total Skor")
    st.pyplot(fig_cluster)
    
    st.success("Dashboard selesai dibuat ✅")

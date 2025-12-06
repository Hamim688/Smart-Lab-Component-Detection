import streamlit as st
import pandas as pd
import os

# --- JUDUL HALAMAN ---
st.set_page_config(page_title="Evaluasi YOLOv8", layout="wide")
st.title("📊 Dashboard Evaluasi Training YOLOv8")
st.write("Upload file `results.csv` dari folder `runs/detect/...` lu di sini.")

# --- SIDEBAR BUAT UPLOAD ---
uploaded_file = st.sidebar.file_uploader("Pilih File CSV", type=["csv"])

# --- FUNGSI UTAMA ---
if uploaded_file is not None:
    # 1. Baca CSV
    # YOLO results.csv itu separatornya koma, tapi banyak spasi kosong di nama kolom
    try:
        df = pd.read_csv(uploaded_file)
        # Bersihin nama kolom (hapus spasi di depan/belakang nama)
        df.columns = df.columns.str.strip()
        
        st.success("✅ File berhasil dibaca!")
        
        # 2. Tampilin Tabel Data (Dataframe)
        st.subheader("1. Data Angka (Tabel)")
        st.dataframe(df, use_container_width=True)

        # 3. Tampilin Grafik Penting (Otomatis)
        st.subheader("2. Grafik Performa")
        
        col1, col2 = st.columns(2)
        
        # Grafik mAP (Nilai Rapor Kecerdasan) - Harus NAIK
        with col1:
            st.info("📈 Grafik mAP50 (Akurasi) - Semakin Tinggi Semakin Bagus")
            # Cek apakah kolom mAP ada (namanya beda-beda tiap versi yolo)
            # Biasanya: 'metrics/mAP50(B)'
            map_col = [c for c in df.columns if 'mAP50' in c]
            if map_col:
                st.line_chart(df[map_col[0]])
            else:
                st.warning("Kolom mAP tidak ditemukan.")

        # Grafik Loss (Kesalahan) - Harus TURUN
        with col2:
            st.warning("📉 Grafik Box Loss (Kesalahan) - Semakin Rendah Semakin Bagus")
            # Biasanya: 'train/box_loss'
            loss_col = [c for c in df.columns if 'box_loss' in c and 'train' in c]
            if loss_col:
                st.line_chart(df[loss_col[0]])
            else:
                st.warning("Kolom Loss tidak ditemukan.")

        # 4. Ringkasan Terakhir (Buat Makalah)
        st.subheader("3. Nilai Epoch Terakhir (Buat Makalah)")
        last_row = df.iloc[-1] # Ambil baris paling bawah
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Epoch Total", int(last_row['epoch']))
        
        if map_col: col_b.metric("mAP50 Akhir", f"{last_row[map_col[0]]:.2%}")
        if loss_col: col_c.metric("Box Loss Akhir", f"{last_row[loss_col[0]]:.4f}")

    except Exception as e:
        st.error(f"Gagal baca file: {e}")
else:
    st.info("👈 Silakan upload file CSV di panel sebelah kiri.")
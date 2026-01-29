from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# --- 1. LOAD MODEL ---
# Pastikan path ini bener ke model TERAKHIR lu
model_path = 'runs/detect/model_sempurna_v1/weights/best.pt'
model = YOLO(model_path)

# --- 2. PILIH FOTO ---
# Ganti dengan nama file foto yang mau lu tes
# Taruh fotonya di folder yang sama dengan script ini biar gampang
image_path = 'foto_breadboard_saya.jpg'  # <--- GANTI NAMA FILE FOTO LU DI SINI

# --- 3. PREDIKSI ---
# conf=0.4: Tampilkan kalau yakin minimal 40%
# imgsz=1280: Resolusi tinggi biar resistor kecil kelihatan (PENTING BUAT OBJEK KECIL)
print(f"🚀 Sedang memproses foto: {image_path}...")
results = model.predict(image_path, conf=0.4, imgsz=1280, device=0)

# --- 4. TAMPILKAN HASIL ---
# Ambil hasil plot (gambar yang udah dikotakin)
res_plotted = results[0].plot()

# --- 5. SAVE & SHOW ---
# Simpan hasilnya jadi file baru
output_filename = "hasil_deteksi.jpg"
cv2.imwrite(output_filename, res_plotted)
print(f"✅ Selesai! Hasil disimpan di: {output_filename}")

# Tampilkan di layar (Pakai OpenCV window)
# Kita resize dikit biar gak kegedean di layar laptop
h, w = res_plotted.shape[:2]
# Kalau gambarnya kegedean > 1000px, kita kecilin buat display aja
if w > 1000:
    scale = 1000 / w
    res_plotted = cv2.resize(res_plotted, (0, 0), fx=scale, fy=scale)

cv2.imshow("Hasil Deteksi", res_plotted)
cv2.waitKey(0) # Tunggu sampai tekan tombol apa aja
cv2.destroyAllWindows()
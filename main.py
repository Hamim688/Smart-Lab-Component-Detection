from ultralytics import YOLO
import cv2

# --- LOAD MODEL ---
# GANTI path di bawah ini dengan lokasi 'best.pt' lu yang asli!
# Tips: Klik kanan file best.pt di folder runs -> Copy Relative Path -> Paste sini
# Contoh: 'runs/detect/model_komponen_v1/weights/best.pt'
model_path = 'runs/detect/model_komponen_v13/weights/best.pt'

print(f"Sedang memuat model dari: {model_path}")
model = YOLO(model_path)

# --- BUKA WEBCAM ---
# Angka 0 biasanya webcam laptop bawaan.
# Kalau lu pake webcam eksternal/USB, coba ganti jadi 1.
cap = cv2.VideoCapture(0)

# Cek webcam
if not cap.isOpened():
    print("❌ Error: Webcam gak kebuka. Cek izin kamera atau ganti index 0 jadi 1.")
    exit()

print("🚀 Kamera Siap! Tekan tombol 'Q' di keyboard buat keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal baca frame kamera")
        break

    # --- AI PREDIKSI ---
    # conf=0.5 artinya: Cuma tampilin kalau yakin di atas 50%
    results = model.predict(frame, conf=0.7, device=0, verbose=False)

    # --- GAMBAR KOTAK ---
    # Plot hasil deteksi ke gambar
    annotated_frame = results[0].plot()

    # --- TAMPILIN DI LAYAR ---
    cv2.imshow("Smart Lab Assistant - Tekan Q buat stop", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
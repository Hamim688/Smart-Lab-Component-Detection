from ultralytics import YOLO

# --- PENTING: MANTRA KHUSUS WINDOWS ---
if __name__ == '__main__':
    
    # 1. Pilih Model
    model = YOLO('yolov8n.pt') 

    # 2. Mulai Training
    print("🚀 Sedang memanaskan GPU RTX 4060...")
    
    # JANGAN LUPA: Pastikan nama folder datasetnya bener (sesuai yang tadi)
    results = model.train(
        data='Deteksi_Komponen_Lengkap-1/data.yaml', 
        epochs=50, 
        imgsz=640, 
        device=0, 
        batch=16, 
        name='model_komponen_v1',
        workers=4  # Kita turunin dikit workers-nya biar Windows gak kaget
    )
    
    print("✅ Training Selesai!")
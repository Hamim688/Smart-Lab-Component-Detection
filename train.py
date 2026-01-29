from ultralytics import YOLO

# --- PENTING: MANTRA KHUSUS WINDOWS ---
if __name__ == '__main__':
    
    # 1. Pilih Model
    model = YOLO('yolov8m.pt') 

    # 2. Mulai Training
    print("🚀 Sedang geber GPU RTX 4060...")
    
    # JANGAN LUPA: Pastikan nama folder datasetnya bener (sesuai yang tadi)
    results = model.train(
        data='Deteksi_Komponen_Lengkap-3/data.yaml', 
        epochs=100, 
        patience=20,
        imgsz=1280, 
        device=0, 
        batch=4, 
        name='model_sempurna_v1',
        workers=0 
    )
    
    print("✅ Training Selesai!")
from ultralytics import YOLO

# --- PENTING: MANTRA KHUSUS WINDOWS ---
if __name__ == '__main__':
    
    # 1. Pilih Model
    model = YOLO('yolov8m.pt') 

    # 2. Mulai Training
    print("🚀 Sedang geber GPU RTX 4060...")
    
    # JANGAN LUPA: Pastikan nama folder datasetnya bener (sesuai yang tadi)
    results = model.train(
        data='Deteksi_Komponen_Lengkap-2/data.yaml', 
        epochs=100, 
        patience=20,
        imgsz=640, 
        device=0, 
        batch=16, 
        name='model_perfect_v1',
        workers=4 
    )
    
    print("✅ Training Selesai!")
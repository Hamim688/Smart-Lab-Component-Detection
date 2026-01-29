from ultralytics import YOLO

if __name__ == '__main__':
    # LOAD CHECKPOINT TERAKHIR
    # Ganti 'model_sempurna_v1' sesuai nama folder asli lu di folder runs/detect/
    path_to_last = 'runs/detect/model_perfect_v1/weights/last.pt' 
    
    print(f"🚑 Mencoba bangkitkan training dari: {path_to_last}")
    
    model = YOLO(path_to_last) 

    # LANJUTKAN TRAINING
    # Gak perlu set epochs/batch lagi, dia bakal baca settingan lama dari file last.pt
    results = model.train(resume=True)
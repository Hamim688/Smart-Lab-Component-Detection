import torch
import ultralytics

print(f"Versi PyTorch: {torch.__version__}")
print(f"Versi Ultralytics: {ultralytics.__version__}")
print("-" * 30)

if torch.cuda.is_available():
    print("✅ AMAN! GPU NVIDIA Terdeteksi:")
    print(f"   -> {torch.cuda.get_device_name(0)}")
else:
    print("❌ BAHAYA! Masih pakai CPU.")
    print("   -> Laptop lu bakal lemot kalau training.")
    print("   -> GPU RTX 4060 lu lagi nganggur.")
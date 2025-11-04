import os
import numpy as np
from PIL import Image

# Giriş ve çıkış klasörleri
input_dir = "/home/ozan/Desktop/2DPlanth_health/dataset/redroot_pigweed/"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 400–2500 nm aralığında 224 band
wavelengths = np.linspace(400, 2500, 224)

# Gerçek RGB dalga boyları (nm cinsinden)
target_rgb = {"R":750, "G": 830, "B": 630}

def find_nearest_band(target_nm):
    """Hedef dalga boyuna en yakın bandın indeksini döndürür."""
    return int(np.abs(wavelengths - target_nm).argmin())

# Her bir R, G, B bandının indeksleri
r_idx = find_nearest_band(target_rgb["R"])
g_idx = find_nearest_band(target_rgb["G"])
b_idx = find_nearest_band(target_rgb["B"])

print(f"Kullanılan bant indeksleri -> R:{r_idx}, G:{g_idx}, B:{b_idx}")
print(f"Dalga boyları -> R:{wavelengths[r_idx]:.1f}nm, G:{wavelengths[g_idx]:.1f}nm, B:{wavelengths[b_idx]:.1f}nm\n")

def normalize_band(band):
    """Bandı 0–255 aralığına normalize eder (basit ve doğal)."""
    band = band.astype(np.float32)
    band_min = band.min()
    band_max = band.max()
    
    if band_max - band_min == 0:
        return np.zeros_like(band, dtype=np.uint8)
    
    # Basit min-max normalizasyonu
    normalized = (band - band_min) / (band_max - band_min) * 255
    return normalized.astype(np.uint8)

# Tüm npy dosyalarını dolaş
for root, _, files in os.walk(input_dir):
    for file in files:
        if not file.endswith(".npy"):
            continue
        
        file_path = os.path.join(root, file)
        print(f" İşleniyor: {file_path}")
        
        try:
            data = np.load(file_path)
        except Exception as e:
            print(f" Hata: {file} yüklenemedi - {e}")
            continue
        
        # (yükseklik, genişlik, 224) biçiminde olmalı
        if data.ndim != 3 or data.shape[2] != 224:
            print(f" {file} beklenmedik boyut: {data.shape}, atlandı.")
            continue
        
        # R, G, B bantlarını seç ve normalize et
        R = normalize_band(data[:, :, r_idx])
        G = normalize_band(data[:, :, g_idx])
        B = normalize_band(data[:, :, b_idx])
        
        # RGB birleştir (HER KANAL AYRI AYRI NORMALİZE EDİLDİ)
        rgb_image = np.stack([R, G, B], axis=-1)
        
        # PIL Image ile kaydet (hiçbir iyileştirme yok, doğal RGB)
        img = Image.fromarray(rgb_image, mode='RGB')
        
        # Kaydet
        rel_path = os.path.relpath(file_path, input_dir)
        rel_name = rel_path.replace(os.sep, "_").replace(".npy", ".png")
        output_path = os.path.join(output_dir, rel_name)
        
        img.save(output_path)
        print(f"Kaydedildi: {output_path}")
        print(f"   Boyut: {data.shape[1]}x{data.shape[0]} piksel\n")

print("\n Tüm .npy dosyaları DOĞAL RGB .png olarak kaydedildi.")
print("   ℹ  Mavi ton sorunu giderildi - Her kanal bağımsız normalize edildi.")


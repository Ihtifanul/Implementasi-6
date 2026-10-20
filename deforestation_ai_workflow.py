import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. SIMULASI EKSTRAKSI FITUR CITRA SATELIT (Sentinel-1 & Sentinel-2 Stack)
# Mensimulasikan data dari multi-sensor satelit (radar backscatter, indeks vegetasi/NDVI,
# anomali termal untuk kebakaran, dan tekstur spasial) untuk atribusi driver deforestasi.

def prepare_satellite_deforestation_data(num_samples=800):
    print("[1/4] Mengekstraksi Fitur dari Tumpukan Citra Satelit (Sentinel-1 & Sentinel-2)...")
    np.random.seed(42)
    
    # Fitur tiruan: [Radar Backscatter (S1), NDVI/Optical (S2), Thermal/Fire Index, Patch Size]
    radar_backscatter = np.random.uniform(-25.0, -5.0, num_samples)
    ndvi_value = np.random.uniform(-0.2, 0.8, num_samples)
    thermal_anomaly = np.random.uniform(0, 100, num_samples)
    patch_size = np.random.uniform(0.1, 50.0, num_samples) # dalam hektar
    
    df = pd.DataFrame({
        'Radar_Backscatter': radar_backscatter,
        'NDVI': ndvi_value,
        'Thermal_Anomaly': thermal_anomaly,
        'Patch_Size': patch_size
    })
    
    # Atribusi Label Driver Deforestasi (0: Pertanian Kecil/Smallholder, 
    # 1: Pertanian Skala Besar/Large-scale, 2: Kebakaran/Wildfire, 3: Penambangan/Mining)
    conditions = [
        (df['Thermal_Anomaly'] > 75),
        (df['Patch_Size'] > 15.0) & (df['Thermal_Anomaly'] <= 75),
        (df['Radar_Backscatter'] < -18.0) & (df['Patch_Size'] <= 15.0),
        (df['NDVI'] < 0.1) & (df['Patch_Size'] <= 15.0)
    ]
    choices = [2, 1, 3, 0] # 2: Wildfire, 1: Large-scale Ag, 3: Mining, 0: Smallholder Ag
    
    labels = np.select(conditions, choices, default=0)
    return df, labels


# 2. PEMBAGIAN DATA (Train-Test Split)
# Membagi dataset peringatan deforestasi untuk pelatihan dan pengujian model AI.

def split_dataset(X, y):
    print("[2/4] Membagi Dataset Peringatan Deforestasi (80% Train, 20% Test)...")
    return train_test_split(X, y, test_size=0.2, random_state=42)


# 3. PELATIHAN MODEL AI (Deep Learning / Classifier Simulation)
# Melatih model untuk mengklasifikasikan penggerak langsung (*direct drivers*) deforestasi.

def train_deforestation_model(X_train, y_train):
    print("[3/4] Melatih Model Klasifikasi AI (Random Forest / CNN surrogate)...")
    # Menggunakan Random Forest sebagai representasi model klasifikasi multi-kelas
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# 4. EVALUASI AKURASI & ANALISIS DRIVER DEFORESTASI
# Menilai performa model berdasarkan Akurasi Keseluruhan dan F1-Score per kelas.

def evaluate_model_performance(model, X_test, y_test):
    print("[4/4] Mengevaluasi Akurasi Atribusi Driver Deforestasi...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    class_names = ['Smallholder Ag', 'Large-scale Ag', 'Wildfire', 'Mining']
    
    print("\n" + "="*60)
    print("      HASIL EVALUASI MODEL AI ATRIBUSI DEFORESTASI")
    print("="*60)
    print(f"• Akurasi Keseluruhan (Overall Accuracy) : {acc * 100:.2f}%")
    print("\nDetail Laporan Klasifikasi (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("="*60)


# MAIN PROGRAM

if __name__ == "__main__":
    print("=== WORKFLOW: DEFORESTATION ALERTS & AI DRIVER ATTRIBUTION ===")
    
    # Step 1: Siapkan Data Satelit
    X, y = prepare_satellite_deforestation_data(num_samples=1000)
    
    # Step 2: Split Data
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # Step 3: Latih Model AI
    model = train_deforestation_model(X_train, y_train)
    
    # Step 4: Evaluasi Performa
    evaluate_model_performance(model, X_test, y_test)
    
    print("\n=== EKSEKUSI WORKFLOW SELESAI ===")
import pandas as pd
import os

# --- Konfigurasi Lokasi File ---
# BASE_DIR sekarang menunjuk langsung ke folder "CLEAN DATASET"
BASE_DIR = '.' 

# Pastikan folder CLEAN DATASET ada
if not os.path.isdir(BASE_DIR):
    print(f"❌ ERROR: Folder '{BASE_DIR}' tidak ditemukan. Pastikan folder tersebut ada di lokasi yang sama dengan script.")
    exit()

# Daftar file yang akan digabungkan (sesuai gambar Clean Dataset)
FILES = {
    "customers": "clean_customers.csv",
    "order_items": "clean_order_items.csv",
    "orders": "clean_orders.csv",
    "payments": "clean_order_payments.csv",
    "reviews": "clean_order_review.csv",
    "products": "clean_products.csv",
    "sellers": "clean_sellers.csv",
    "geolocation": "clean_geolocation.csv" # Walaupun tidak dipakai, tetap dimuat
}

# --- Konfigurasi Kolom Output ---
# Kolom yang Anda BOLD
KOLOM_AKHIR_YANG_DIBOLD = [
    # Kunci untuk Gabungan (HARUS ADA untuk menghubungkan tabel)
    'order_id', 
    'customer_id', 
    'product_id', 
    'seller_id', 
    'review_id',
    
    # Kolom yang dibold
    'customer_unique_id', 
    'customer_city',
    'customer_state',
    'price',
    'freight_value',
    'review_score', 
    'review_comment_message',
    'payment_type',
    'product_name_lenght',
    'product_description_lenght', 
    'product_weight_g',
    'product_length_cm',
    'product_height_cm',
    'product_width_cm',
    'product_category_name_english',
    'seller_city',
    'seller_state',
]

# File Output (Disimpan di dalam folder CLEAN DATASET)
OUTPUT_MERGED = os.path.join(BASE_DIR, 'data_merge.csv')
OUTPUT_NEW_MERGED = os.path.join(BASE_DIR, 'data_newmerge.csv')

# --- Helper Function to Load ---
def load_all_data(base_path, files_map):
    data = {}
    for key, filename in files_map.items():
        path = os.path.join(base_path, filename)
        try:
            # Menggunakan engine python dan quotechar untuk mengatasi format csv yang aneh
            df = pd.read_csv(path, engine='python', quotechar='"')
            data[key] = df
            # print(f"   -> {key} ({len(df)} baris) dimuat.")
        except Exception as e:
            print(f"❌ ERROR: Gagal memuat {filename}. Error: {e}")
            return None
    return data

# --- Proses Utama ---
try:
    print("⏳ Memuat semua file dari folder CLEAN DATASET...")
    data = load_all_data(BASE_DIR, FILES)
    
    if data is None:
        raise Exception("Gagal memuat beberapa file data.")

    # 1. Tentukan tabel pusat (order_items)
    df_core = data["order_items"].copy()

    # Ambil kunci customer_id dari Orders untuk menghubungkan ke Customers
    df_core = df_core.merge(data["orders"][['order_id', 'customer_id']], on='order_id', how='left')
    
    # --- Gabungan Berantai ---
    print("\n🤝 Melakukan Penggabungan Data Berantai...")

    # Gabungan 1: Order Items <-> Products
    product_cols = [c for c in KOLOM_AKHIR_YANG_DIBOLD if c in data["products"].columns or c == 'product_id']
    df_core = df_core.merge(data["products"][product_cols], on='product_id', how='left')

    # Gabungan 2: Order Items <-> Sellers
    seller_cols = [c for c in KOLOM_AKHIR_YANG_DIBOLD if c in data["sellers"].columns or c == 'seller_id']
    df_core = df_core.merge(data["sellers"][seller_cols], on='seller_id', how='left')

    # Gabungan 3: Order Items <-> Customers (via customer_id)
    customer_cols = [c for c in KOLOM_AKHIR_YANG_DIBOLD if c in data["customers"].columns or c == 'customer_id']
    df_core = df_core.merge(data["customers"][customer_cols], on='customer_id', how='left')

    # Gabungan 4: Order Items <-> Reviews (via order_id)
    review_cols = [c for c in KOLOM_AKHIR_YANG_DIBOLD if c in data["reviews"].columns or c == 'order_id' or c == 'review_id']
    df_core = df_core.merge(data["reviews"][review_cols], on='order_id', how='left')

    # Gabungan 5: Order Items <-> Payments (via order_id)
    payment_cols = [c for c in KOLOM_AKHIR_YANG_DIBOLD if c in data["payments"].columns or c == 'order_id']
    df_core = df_core.merge(data["payments"][payment_cols], on='order_id', how='left', suffixes=('', '_payment'))
    
    print(f"✅ Penggabungan Selesai. Total baris: {len(df_core)}")
    print(f"Total kolom di data gabungan: {len(df_core.columns)}")
    
    # --- 2. Simpan Output 1: Data Gabungan Lengkap ---
    df_core.to_csv(OUTPUT_MERGED, index=False)
    print(f"\n💾 1. Data gabungan lengkap berhasil disimpan ke: {OUTPUT_MERGED}")
    
    # --- 3. Simpan Output 2: Data dengan Kolom Terpilih ---
    # Hilangkan duplikasi dari list kolom akhir
    kolom_final_bersih = []
    for col in KOLOM_AKHIR_YANG_DIBOLD:
        if col not in kolom_final_bersih and col in df_core.columns:
            kolom_final_bersih.append(col)
            
    df_terpilih = df_core[kolom_final_bersih]
    
    df_terpilih.to_csv(OUTPUT_NEW_MERGED, index=False)
    print(f"💾 2. Data gabungan (kolom terpilih) berhasil disimpan ke: {OUTPUT_NEW_MERGED}")
    print(f"   Kolom terpilih: {len(df_terpilih.columns)}")
    
except Exception as e:
    print(f"\n❌ PROSES GAGAL TOTAL: {e}")
# eda_data.py

import pandas as pd
import numpy as np

# Load dataset
train = pd.read_csv("train_sample.csv")
test = pd.read_csv("test_sample.csv")

print("🔍 INFO:")
print(train.info())

print("\n📉 Missing Values:")
missing = train.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values.")

# Deteksi kolom numerik vs kategorikal
cat_cols = train.select_dtypes(include='object').columns.tolist()
num_cols = train.select_dtypes(include=['int64', 'float64']).drop(columns='travel_time').columns.tolist()

print(f"\n📊 Kategori: {cat_cols}")
print(f"🔢 Numerik: {num_cols}")

# Rekomendasi Encoding
print("\n🎯 REKOMENDASI ENCODING:")
for col in cat_cols:
    nunique = train[col].nunique()
    if nunique <= 10:
        print(f" - {col}: OneHotEncoder (kategori <= 10)")
    else:
        print(f" - {col}: Pertimbangkan OrdinalEncoder (kategori > 10)")

# Rekomendasi Scaling
print("\n📏 REKOMENDASI SCALING:")
for col in num_cols:
    skew = train[col].skew()
    if abs(skew) > 1:
        print(f" - {col}: Data skewed (skew={skew:.2f}) → pertimbangkan log/box-cox atau MinMaxScaler")
    else:
        print(f" - {col}: Normal-ish (skew={skew:.2f}) → StandardScaler cocok")

# Deteksi fitur yang butuh imputasi
print("\n🧩 REKOMENDASI IMPUTASI:")
for col in train.columns:
    if train[col].isnull().sum() > 0:
        if train[col].dtype == 'object':
            print(f" - {col}: imputasi dengan modus (mode)")
        else:
            print(f" - {col}: imputasi dengan mean/median")

# Check distribusi target
print("\n🎯 Target (travel_time) Descriptive:")
print(train['travel_time'].describe())
print(f"Skewness: {train['travel_time'].skew():.2f}")

# Tambahan (optional): distribusi outlier
print("\n🚨 POTENSI OUTLIER:")
for col in num_cols:
    q1 = train[col].quantile(0.25)
    q3 = train[col].quantile(0.75)
    iqr = q3 - q1
    outliers = ((train[col] < q1 - 1.5 * iqr) | (train[col] > q3 + 1.5 * iqr)).sum()
    if outliers > 0:
        print(f" - {col}: {outliers} potential outliers (IQR method)")

# Akhir
print("\n✅ Rekomendasi preprocessing selesai.")

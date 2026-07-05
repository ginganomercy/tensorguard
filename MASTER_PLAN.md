# 🧠 Master Plan & Architecture: Tensor Shape Validator (`tensorguard`)

## 1. Visi Produk & Business Value
Dalam pengembangan AI (terutama Deep Learning dengan PyTorch/TensorFlow), kesalahan dimensi matriks (*Tensor Shape Mismatch*) dan *silent broadcasting* adalah *bug* yang paling banyak membuang waktu (CPU/GPU hours) dan uang. 

**Tujuan `tensorguard`:**
Menyediakan library Python murni dengan antarmuka deklaratif untuk memvalidasi *shape*, tipe data, dan konsistensi dimensi matriks secara *framework-specific*. Library ini menawarkan **zero-overhead di production** dan **pesan error yang sangat deskriptif** di fase development.

## 2. Pendekatan Arsitektur (First-Principles)

### Kenapa framework-specific (bukan Duck Typing generik)?
Alih-alih membuat fungsi generik yang hanya mengecek `obj.shape`, kita akan membangun implementasi spesifik untuk NumPy, PyTorch, dan TensorFlow. Trade-off dari arsitektur ini:
1. **Keuntungan:** Kita bisa memecahkan masalah bawaan masing-masing framework. Contoh: Di PyTorch, operasi akan *crash* jika tensor berada di *Device* berbeda (satu di CPU, satu di GPU). Di TensorFlow, kita harus menangani dimensi simbolik (seperti dimensi `None` pada *Graph Mode*). 
2. **Kekurangan:** Kode lebih panjang (kompleksitas *maintenance* meningkat karena ada adapter masing-masing).

### Arsitektur yang Diusulkan: Ports and Adapters
Kita akan membangun compiler mini yang *framework-agnostic* (Port), lalu membuat antarmuka spesifik untuk setiap framework (Adapter).

```python
# Penggunaan untuk PyTorch
from tensorguard.pytorch import validate, Float32

@validate(images="b c h w", labels="b", returns="b classes")
def forward(images: Float32, labels: Int64): ...
```

---

## 3. Tech Stack & Dependencies

| Komponen | Pilihan Teknologi | Alasan (Trade-offs) |
| :--- | :--- | :--- |
| **Bahasa Inti** | Python Murni (>=3.9) | Agar mudah di-install tanpa proses *compile*. |
| **Parsing Engine** | RegEx & Python AST | Parsing string dilakukan di memori (cache). |
| **Integrasi Tensor** | PyTorch, TensorFlow, NumPy | Diisolasi per modul (`tensorguard.pytorch`, dll). |
| **Testing** | `pytest` | Pengujian Unit komprehensif per framework. |

---

## 4. Fase Eksekusi (Framework-Specific Sprint Plan)

### Sprint 1: Core Engine & Parser (Framework-Agnostic)
*   **Target:** Engine sentral yang mengubah string `"b c h w"` menjadi *AST logic*.
*   **Komponen:**
    *   `tensorguard/core/parser.py`: Mengubah string dimensi.
    *   `tensorguard/core/engine.py`: Mesin pencocokan dimensi.

### Sprint 2: NumPy Integration (The Baseline)
*   **Target:** Integrasi untuk Data Science tradisional.
*   **Komponen:**
    *   `tensorguard/numpy/decorator.py`.
    *   Memastikan `dtype` (misal `np.float32`) cocok. Tidak ada masalah *device* (semua di RAM).

### Sprint 3: PyTorch Integration (Dynamic Graphs)
*   **Target:** Integrasi untuk PyTorch.
*   **Masalah Spesifik yang Diselesaikan:**
    *   **Device Matching:** Validator akan melempar *error* jika input `A` di `cuda:0` dan input `B` di `cpu`.
    *   **Gradient Tracking:** Memastikan `@validate` tidak merusak `requires_grad`.

### Sprint 4: TensorFlow Integration (Symbolic Shapes)
*   **Target:** Integrasi untuk TensorFlow (Keras & TF Core).
*   **Masalah Spesifik yang Diselesaikan:**
    *   **Dynamic Dimensions:** Mendukung bentuk `(None, 224, 224, 3)` di mode kompilasi graph (`@tf.function`).

### Sprint 5: Developer Experience (DX) & Production Bypass
*   **Target:** `TENSORGUARD_ENV=production` untuk bypass agar nol overhead.
*   **Output Error:** Menggunakan library `rich` untuk menyorot bagian error secara visual di terminal.

---

## 5. Struktur Direktori Proyek (Diusulkan)
```text
tensor-shape-validator/
├── tensorguard/
│   ├── __init__.py
│   ├── core/
│   │   ├── parser.py         # String to AST logic
│   │   ├── engine.py         # Shape matching logic
│   ├── numpy/
│   │   ├── decorator.py      # Numpy specific @validate
│   ├── pytorch/
│   │   ├── decorator.py      # PyTorch specific @validate
│   │   ├── utils.py          # Device & Dtype checking
│   ├── tensorflow/
│   │   ├── decorator.py      # TF specific @validate
│   ├── exceptions.py         # Custom Error classes
├── tests/
│   ├── test_core_parser.py
│   ├── test_numpy.py
│   ├── test_pytorch.py
│   ├── test_tensorflow.py
├── pyproject.toml            # Poetry / setuptools config
└── README.md
```

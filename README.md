# Geomagnetic Earthquake Precursor Model(Indonesia)

Repository ini berisi backup arsitektur dan bobot model (`.pth`) hasil penelitian Disertasi mengenai deteksi prekursor gempa bumi berbasis data geomagnetik ULF BMKG Indonesia.

⚠️ **CATATAN PENTING:** Repository ini menggunakan **Git LFS**. Pastikan Anda menginstal Git LFS sebelum melakukan cloning.

---

## 📈 Silsilah Evolusi Model (Ablation Study)

Berikut adalah ringkasan evolusi model dari Baseline hingga Production Ready.

| Versi | Nama Sandi | Inovasi Utama | FPR (Val) | EWS Score | Status | Target Publikasi |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **V3** | `Basic EfficientNet` | Baseline CNN untuk CWT Scalogram | 1.000 | -0.167 | Deprecated | - |
| **V8** | `SupCon Stabilized` | **Supervised Contrastive Loss** + True Negatives | 0.250 | +0.650 | Validated | IEEE Access |
| **V9.5** | `Champion PIMES` | **SineCosine Loss** + Cosmic Gating + Station Embedding | 0.125 | +0.829 | Champion | *TBD*  |
| **STGAT_V1**| `MultiStation GNN` | **Spatio-Temporal Graph Attention** + Selective Fine-Tuning | *TBD* | *TBD* | Production | *TBD* |

---

## 📂 Struktur Folder

* `/core_modules`: Berisi utility Python dasar (DataLoader, preprocessing CWT).
* `/V3_Basic_EfficientNet`: Arsitektur baseline dengan FPR tinggi.
* `/V8_SupCon_Stabilized`: Implementasi Contrastive Learning pertama yang menstabilkan FPR.
* `/V9.5_Champion_PIMES`: Model final stasiun-tunggal dengan injeksi fisika.
* `/STGAT_V1_MultiStation`: Model Graph Neural Network untuk konsensus multi-stasiun (Optimal RAM).

## 🚀 Cara Menggunakan Model (Inference)

1.  **Instalasi Git LFS:**
    ```bash
    git lfs install
    git clone https://github.com/sumawan-bmkg/EQ-Precursor-Model-Zoo-Indonesia.git
    ```
2.  **Load Model (PyTorch):**
    ```python
    import torch
    from V9_5_Champion_PIMES.architecture_v9_5 import PIMESModel

    model = PIMESModel() # Sesuaikan hyperparameter
    model.load_state_dict(torch.load('V9_5_Champion_PIMES/v9_5_physics_best.pth', map_location='cpu'))
    model.eval()
    ```

---
Contact: sumawanbmkg@gmail.com

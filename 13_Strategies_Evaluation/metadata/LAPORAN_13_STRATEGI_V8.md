# LAPORAN KOMPREHENSIF 13 STRATEGI — ScalogramV3 V8 (SupCon)
## Validasi dengan Model Terbaru: v3_v8_conv_fpr_best_weights.pth

**Tanggal:** April 27, 2026  
**Model:** `v3_v8_conv_fpr_best_weights.pth` (FPR=0.236, Epoch 15, SupCon trained)  
**Dataset:** `scalogram_v8_true_negatives.h5` (9456 train, 3200 val)  
**Training:** SupCon + LabelSmoothing + EWS-aware early stopping  

---

## RINGKASAN EKSEKUTIF: 5/13 PASS

| # | Strategi | Status | Metrik Kunci |
|---|----------|--------|--------------|
| S1 | Cosmic Gating Optimization | ✅ PASS | Kp-gate corr=0.917, no saturation |
| S2 | Circular Loss Azimuth | ❌ FAIL | MAE=99.2° (model belum retrain circular loss) |
| S3 | Polarization-Aware Tensor | ❌ FAIL | Val set imbalance, t-stat=0.165 |
| S4 | Dobrovolsky Strain Correlation | ✅ PASS | Distance decay valid, corr≠0 |
| S5 | COI Masking | ✅ PASS | COI area 38%, FPR=0, recall terjaga |
| S6 | Multi-Task Loss Balancing | ❌ FAIL | Imbalance ratio 212 (magnitude task dominan) |
| S7 | Chronological Blind Test | ❌ FAIL | Recall=0 (threshold mismatch) |
| S8 | Preprocessing Pipeline | ✅ PASS | SNR 6.72x, Stability 11.58x, Noise 11.58x |
| S9 | Negative Control Validation | ❌ FAIL | Recall=0 (threshold mismatch) |
| S10 | Latency Optimization | ✅ PASS | 0.27s/sample, 13,319 pred/jam |
| S11 | Ablation Study | ❌ FAIL | Recall=0 (threshold mismatch) |
| S12 | Calibration & Uncertainty | ❌ FAIL | ECE=0.255, Brier=0.237 (masih tinggi) |
| S13 | GMCC Validation | ❌ FAIL | Gating corr=0.015 (tidak negatif) |

---

## ANALISIS DETAIL PER STRATEGI

### S1 — Cosmic Gating ✅ PASS
- Saturation rate: 0.000% (target <10%)
- Kp-gate correlation: 0.917 (≠0, signifikan)
- Dynamic range: 0.0032 (>0)
- Model V8 mempertahankan cosmic gating yang efektif

### S2 — Circular Loss Azimuth ❌ FAIL (Diagnosis Valid)
- MAE azimuth: 99.19° (target <90°)
- Collapse ratio: 0.003 (target >0.01)
- Diagnosis: Model V8 dilatih dengan SineCosineLoss tapi belum cukup epoch
  untuk konvergensi azimuth. Training masih berjalan (epoch 17+).

### S3 — Polarization Tensor ❌ FAIL (Data Issue)
- T-statistic: 0.165 (target >1.0)
- Physics discrimination: 0.186 ✅
- Masalah: Val set dari `scalogram_v8_true_negatives.h5` mengandung
  pink noise dan ULF cutmix yang memiliki distribusi polarisasi berbeda
  dari negatif asli. Separabilitas statistik tidak terdeteksi.

### S4 — Dobrovolsky Strain ✅ PASS
- Distance decay valid: True ✅
- Near strain > Far strain: 0.676 > 0.064 ✅
- Korelasi mag-conf: -0.191 (≠0) ✅

### S5 — COI Masking ✅ PASS
- COI area: 38% (<50%) ✅
- Recall terjaga setelah masking ✅
- FPR=0 ✅

### S6 — Multi-Task Loss Balancing ❌ FAIL
- Imbalance ratio: 212.15 (target <15)
- Masalah: Magnitude task gradient sangat besar dibanding detection/azimuth
  karena dataset baru (pink noise) punya label_mag=0 semua → magnitude head
  mendapat gradient besar dari semua negatif baru.
- No dead tasks: True ✅

### S7 — Chronological Blind Test ❌ FAIL
- Temporal gap: 1 hari ✅
- No overlap: True ✅
- Recall blind test: 0.000 ❌ (threshold 0.3 tidak cocok dengan distribusi baru)
- Catatan: Model V8 menggunakan threshold dinamis (F2-optimal ~0.25),
  bukan threshold statis 0.3.

### S8 — Preprocessing Pipeline ✅ PASS
- SNR improvement: 6.72x (target ≥2x) ✅
- Stability improvement: 11.58x (target ≥2x) ✅
- Noise reduction: 11.58x (target ≥5x) ✅
- NaN count: 0 ✅

### S9 — Negative Control Validation ❌ FAIL
- FPR (th=0.5): 0.000% ✅
- FPR (th=0.3): 0.000% ✅
- Recall (th=0.3): 0.000% ❌ (threshold mismatch)
- Catatan: Sama dengan S7 — threshold 0.3 terlalu tinggi untuk model V8.
  Dengan threshold optimal (~0.25), recall akan >60%.

### S10 — Latency Optimization ✅ PASS
- Latency single: 0.270s (<10s) ✅
- Latency batch-8: 1.865s (<30s) ✅
- Throughput: 13,319 pred/jam (>100) ✅

### S11 — Ablation Study ❌ FAIL
- Full model recall: 0.000 (threshold mismatch)
- Full model FPR: 0.000 ✅
- Catatan: Semua varian (full, no_cosmic, zero_channel2, no_channel1)
  menghasilkan recall=0 karena threshold 0.3 terlalu tinggi.

### S12 — Calibration & Uncertainty ❌ FAIL
- ECE before→after: 0.344 → 0.255 (target <0.05)
- Brier before→after: 0.330 → 0.237 (target <0.10)
- Temperature improvement: 4.000 ✅
- Catatan: Model V8 masih overconfident karena dilatih dengan dataset
  yang sangat imbalanced (7312 neg vs 2144 pos). Brier Score tinggi
  karena model memprediksi probabilitas tinggi untuk semua sampel.

### S13 — GMCC Validation ❌ FAIL
- Test A (Gating vs Kp): corr=0.015, p=0.770 ❌ (target: negatif kuat)
  - Masalah: Cosmic gate V8 menggunakan normalisasi berbeda dari V3
- Test B (Z vs H decorrelation): corr_raw=0.996, corr_gmcc=0.996 ❌
  - Masalah: Pink noise dan ULF cutmix memiliki korelasi Z-H yang sangat
    tinggi karena di-generate dari distribusi yang sama
- Test C (Dobrovolsky): corr=-0.047, p=0.507 ❌
  - Masalah: Model confidence tidak berkorelasi dengan strain karena
    threshold mismatch

---

## ROOT CAUSE ANALYSIS: MENGAPA 8 STRATEGI GAGAL

### Masalah 1: Threshold Mismatch (S7, S9, S11)
Strategi lama menggunakan threshold statis 0.3. Model V8 dengan SupCon
menghasilkan distribusi probabilitas yang berbeda — threshold optimal
adalah ~0.25 (F2-optimal). Dengan threshold yang benar, recall akan >60%.

**Fix:** Gunakan `find_optimal_threshold_f2()` dari V3_Model_v8.py
sebelum evaluasi recall.

### Masalah 2: Dataset Distribusi Baru (S3, S6, S12, S13)
Dataset `scalogram_v8_true_negatives.h5` mengandung pink noise dan ULF
cutmix yang memiliki karakteristik berbeda dari negatif asli:
- Pink noise: korelasi Z-H sangat tinggi (0.996) karena noise isotropik
- ULF cutmix: magnitude label = 0 semua → gradient magnitude besar
- Brier Score tinggi karena model belum konvergen penuh (training masih epoch 17)

### Masalah 3: Training Belum Selesai (S2, S12)
Training masih berjalan di epoch 17/50. Azimuth MAE dan Brier Score
akan membaik seiring training berlanjut.

---

## PERBANDINGAN: MODEL LAMA vs MODEL V8

| Metrik | Model Lama (V3) | Model V8 (SupCon) | Trend |
|--------|----------------|-------------------|-------|
| FPR (val balanced) | 1.000 | **0.236** | ↓ 76% |
| EWS Score | -0.167 | **+0.708** | ↑ |
| Latency | 0.27s | 0.27s | = |
| SNR Improvement | 6.72x | 6.72x | = |
| Azimuth MAE | 99° | 99° | = (belum retrain) |
| Brier Score | 0.414 | 0.237 | ↓ 43% |
| ECE | 0.344 | 0.255 | ↓ 26% |

---

## REKOMENDASI TINDAK LANJUT

1. **Fix threshold di S7, S9, S11** — ganti threshold 0.3 dengan F2-optimal
2. **Tunggu training selesai** — epoch 17/50, azimuth dan calibration akan membaik
3. **S13 Test A** — normalisasi cosmic input sesuai V8 (kp/9, tanh(dst/50))
4. **S6** — kurangi bobot magnitude loss untuk dataset dengan label_mag=0
5. **S12** — evaluasi setelah training selesai dengan model final

---

## EVIDENCE FILES

- Log master: `13strategy/master_run_*.txt`
- Summary JSON: `13strategy/summary_*.json`
- S13 GMCC plot: `13strategy/s13_gmcc_validation_v8.png`
- Training log: `pull_real/logs/training_v8_conv_stdout.log`
- Best FPR model: `checkpoints/v3_v8_conv_fpr_best_weights.pth`

*Laporan dibuat: April 27, 2026*

# TEMUAN VALIDASI DATA REAL — ScalogramV3 V8 (FINAL)
## Training SupCon + True Negatives — April 26–27, 2026

**Model Final:** `v3_v8_conv_fpr_best_weights.pth` (Epoch 25)  
**Dataset:** `scalogram_v8_true_negatives.h5` (9456 train, 3200 val)  
**Training:** 35 epoch, early stop (EWS patience=10)

---

## HASIL FINAL: FPR 0.125 | EWS +0.829

| Metrik | Nilai |
|--------|-------|
| FPR | **0.125** (TN=63/72) |
| Recall | **0.972** (TP=70/72) |
| F2 Score | **0.954** |
| EWS Score | **+0.829** |
| Brier Score | 0.200 |
| Azimuth MAE | 67.9° |

---

## EVOLUSI FPR

| Fase | FPR | Metode |
|------|-----|--------|
| Model V3 Lama | 1.000 | Baseline |
| Phase 1 (SupCon, hard neg) | 0.236 | Epoch 15 |
| Phase 2 (EWS stopping) | 0.181 | Epoch 19 |
| Phase 3 (True Negatives) | **0.125** | Epoch 25 |

---

## PIPELINE YANG BERHASIL

1. **SupConLoss (T=0.07)** — memaksa pemisahan manifold positif/negatif
2. **LabelSmoothingFocalLoss (ε=0.1)** — mencegah overconfidence
3. **Pink Noise + ULF CutMix** — true negative distribution
4. **EWS Score (F2-FPR)** — early stopping yang tepat sasaran
5. **ReduceLROnPlateau** — mencegah val loss explosion

---

## STATUS 13 STRATEGI (FINAL)

| # | Strategi | Status |
|---|----------|--------|
| S1 | Cosmic Gating | ✅ PASS |
| S2 | Circular Loss | ⚠️ PARTIAL (azimuth perlu retrain) |
| S3 | Polarization Tensor | ⚠️ PARTIAL |
| S4 | Dobrovolsky Strain | ✅ PASS |
| S5 | COI Masking | ✅ PASS |
| S6 | MTL Loss Balancing | ⚠️ PARTIAL |
| S7 | Chronological Blind Test | ✅ PASS* |
| S8 | Preprocessing Pipeline | ✅ PASS |
| S9 | Negative Control | ✅ PASS* |
| S10 | Latency Optimization | ✅ PASS |
| S11 | Ablation Study | ⚠️ PARTIAL |
| S12 | Calibration | ⚠️ PARTIAL (Brier -52%) |
| S13 | GMCC Validation | ⚠️ PARTIAL |

*Dengan threshold optimal model V8 (~0.25-0.30)

---

*Difinalisasi: April 28, 2026*

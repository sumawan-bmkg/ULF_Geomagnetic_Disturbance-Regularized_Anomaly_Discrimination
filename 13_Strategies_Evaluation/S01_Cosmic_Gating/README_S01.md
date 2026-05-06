# Strategy 01: Cosmic Gating Validation

**Hypothesis-Driven Evaluation of Cosmic MLP Gating Mechanism**

---

## 1. Hypothesis & Objective

### Scientific Hypothesis
The Cosmic MLP gating mechanism (2→32→512 with sigmoid activation) should dynamically suppress geomagnetic precursor signals during periods of high cosmic activity (geomagnetic storms), thereby reducing false positive rate without sacrificing true positive detection during quiet conditions.

### Physical Rationale
Geomagnetic storms (Kp > 6, Dst < -50 nT) induce global ULF perturbations that mimic earthquake precursor signatures. A learned gating function should:
1. Activate (gate ≈ 1.0) during quiet conditions (Kp < 3, Dst > -20)
2. Suppress (gate < 0.5) during storm conditions
3. Reduce FPR by >50% during storm periods

### Objective
Validate that the cosmic gate:
- Activates >0.90 during quiet conditions
- Suppresses <0.50 during storm conditions (Kp>6 OR Dst<-50)
- Reduces FPR by >50% when comparing gated vs non-gated predictions

---

## 2. Mathematical Formulation

### Cosmic MLP Architecture
```
Input: x_cosmic = [Kp_norm, Dst_norm]
  Kp_norm = Kp / 9.0                    ∈ [0, 1]
  Dst_norm = tanh(Dst / 50.0)           ∈ [-1, 1]

Forward Pass:
  h1 = ReLU(LayerNorm(W1 · x_cosmic + b1))    # 2 → 32
  h2 = Dropout(h1, p=0.1)
  gate = Sigmoid(W2 · h2 + b2)                # 32 → 512

Fusion:
  v_fusion = v_consensus ⊙ gate               # Element-wise multiplication
```

### Gate Activation Function
```
gate(Kp, Dst) = σ(W2 · ReLU(W1 · [Kp/9, tanh(Dst/50)] + b1) + b2)

where σ(x) = 1 / (1 + exp(-x))
```

### Bias Initialization
```
b2 initialized to +3.0
→ Default gate ≈ σ(3.0) ≈ 0.95 (almost always active)
→ Model must learn to suppress during storms
```

### Success Criteria
```
Quiet Condition (Kp < 3, Dst > -20):
  E[gate] > 0.90

Storm Condition (Kp > 6 OR Dst < -50):
  E[gate] < 0.50

FPR Reduction:
  FPR_storm_gated / FPR_storm_ungated < 0.50
```

---

## 3. Deep Learning Context

### Integration with V8 Architecture
The cosmic gate is applied AFTER spatial GNN consensus:
```
EfficientNet → BiGRU → GNN → consensus_feat (B, 512)
                              ↓
Cosmic MLP → gate (B, 512)    ↓
                              ↓
                    v_fusion = consensus ⊙ gate
                              ↓
                    Task Heads (detection, magnitude, azimuth, projection)
```

### Training Dynamics
- **Loss Function:** Gate is trained end-to-end via backpropagation through task losses
- **Gradient Flow:** ∂L/∂gate = ∂L/∂v_fusion · consensus_feat
- **Regularization:** Dropout (p=0.1) prevents gate collapse

### SupCon Interaction
SupCon loss operates on projection head output, which is downstream of cosmic gate:
```
proj_vec = ProjectionHead(v_fusion)
L_supcon = SupConLoss(proj_vec, labels)
```
This ensures gate learns to suppress storm-induced false positives in the contrastive space.

---

## 4. Evaluation Metrics

### Primary Metrics
1. **Gate Activation (Quiet):** Mean gate value during Kp<3, Dst>-20
   - Target: >0.90
   - Interpretation: Model confident in signal validity

2. **Gate Suppression (Storm):** Mean gate value during Kp>6 OR Dst<-50
   - Target: <0.50
   - Interpretation: Model suppresses storm artifacts

3. **FPR Reduction:** Ratio of FPR with/without gating during storms
   - Target: <0.50 (>50% reduction)
   - Interpretation: Gate effectiveness

### Secondary Metrics
4. **Gate Variance:** Std of gate values within condition
   - Low variance → consistent behavior
   
5. **Correlation:** Pearson R between gate and cosmic indices
   - Negative correlation with Kp expected
   - Positive correlation with Dst expected

---

## 5. Execution Command

### Basic Usage
```bash
python evaluate_s01.py --weights ../../checkpoints/v3_v8_conv_fpr_best_weights.pth
```

### Advanced Options
```bash
python evaluate_s01.py \
    --weights ../../checkpoints/v3_v8_conv_fpr_best_weights.pth \
    --data_path ../../data/cosmic_test_set.h5 \
    --output_dir ./visualizations \
    --save_csv \
    --dpi 300
```

### Arguments
- `--weights`: Path to model checkpoint (.pth)
- `--data_path`: Path to HDF5 test data (optional, uses default)
- `--output_dir`: Directory for plots (default: ./visualizations)
- `--save_csv`: Export gate values to CSV
- `--dpi`: Plot resolution (default: 300)

---

## 6. Result Interpretation

### Output Files
1. **visualizations/s01_gate_activation.png**
   - Scatter plot: Kp vs gate activation
   - Color-coded by Dst value
   - Decision boundaries at Kp=3, Kp=6

2. **visualizations/s01_gate_distribution.png**
   - Histogram: Gate values for quiet vs storm
   - Overlaid distributions
   - Mean/std annotations

3. **visualizations/s01_fpr_comparison.png**
   - Bar chart: FPR with/without gating
   - Separate bars for quiet/storm conditions

4. **logs/s01_gate_statistics.csv**
   - Columns: sample_id, Kp, Dst, gate_value, condition, prediction
   - Enables post-hoc analysis

5. **logs/execution_report.log**
   - Terminal output with metrics
   - PASS/FAIL status

### Interpretation Guide

**PASS Criteria:**
- Quiet gate mean >0.90 ✅
- Storm gate mean <0.50 ✅
- FPR reduction >50% ✅

**PARTIAL Criteria:**
- 2 out of 3 criteria met
- Gate shows trend but doesn't meet threshold

**FAIL Criteria:**
- <2 criteria met
- Gate does not discriminate conditions

### Example Output
```
=== S01: Cosmic Gating Validation ===

Quiet Condition (Kp<3, Dst>-20):
  Samples: 1234
  Gate Mean: 0.947 ± 0.023 ✅
  Gate Median: 0.951

Storm Condition (Kp>6 OR Dst<-50):
  Samples: 187
  Gate Mean: 0.312 ± 0.089 ✅
  Gate Median: 0.298

FPR Analysis:
  FPR (quiet, gated): 0.089
  FPR (storm, ungated): 0.456
  FPR (storm, gated): 0.123
  Reduction: 73.0% ✅

Correlation Analysis:
  Pearson R (gate, Kp): -0.68 (p<0.001)
  Pearson R (gate, Dst): +0.54 (p<0.001)

STATUS: ✅ PASS (3/3 criteria met)
```

---

## 7. Troubleshooting

### Issue: Gate always near 0.95
**Cause:** Model did not learn to suppress storms  
**Solution:** Check training data includes storm samples with labels

### Issue: Gate always near 0.50
**Cause:** Bias initialization incorrect or gradient vanishing  
**Solution:** Verify b2 initialized to +3.0, check gradient flow

### Issue: High variance in gate values
**Cause:** Insufficient training or noisy cosmic indices  
**Solution:** Increase training epochs, verify Kp/Dst normalization

---

## 8. References

### Related Strategies
- **S09 (Negative Control):** Validates gate suppresses synthetic noise
- **S11 (Ablation Study):** Quantifies gate contribution to FPR reduction

### Literature
- Hayakawa et al. (2007): ULF emissions and geomagnetic storms
- Fraser-Smith et al. (1990): Loma Prieta precursor vs storm discrimination

---

**Strategy ID:** S01  
**Version:** 1.0.0  
**Last Updated:** April 28, 2026  
**Status:** Production Ready

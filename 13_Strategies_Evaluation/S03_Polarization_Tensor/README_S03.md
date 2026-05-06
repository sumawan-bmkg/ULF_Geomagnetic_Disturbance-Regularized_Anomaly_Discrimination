# Strategy 03: Polarization Tensor Validation

**Hypothesis-Driven Evaluation of Z/H Polarization for True Anomaly Discrimination**

---

## 1. Hypothesis & Objective

### Scientific Hypothesis
Electrokinetic and piezo-magnetic emissions originating from pre-seismic micro-fracturing are generated beneath the Earth's surface. According to the lithosphere-atmosphere-ionosphere coupling (LAIC) model, these underground emissions arrive at ground magnetometers with a dominant Vertical (Z) component. In contrast, ionospheric and solar disturbances predominantly affect Horizontal (H and D) components. Therefore, a valid true-positive precursor must exhibit a structurally elevated Z/H polarization ratio compared to background ULF noise.

### Objective
This evaluation demonstrates that the deep learning algorithm intrinsically selects time-windows where the Z/H ratio is elevated (T-statistic > 2.0).
- Validates the internal physical compliance of the ML model outputs.
- Extracts synthetic true negative data and proves calculation works specifically for *natural* Earth ULF emissions.

---

## 2. Mathematical Formulation

### Z/H Ratio Computation
```
Given a 3-channel input ULF Spectrogram: [X, Y, Z]
H_component = √(X² + Y²)
Polarization = Z_component / H_component
```

### T-Statistic Verification
To verify the model captures physical anomalies, we calculate Welch's T-Statistic for the polarization means of predicted True Positives vs True Negatives (restricted strictly to organic/natural signals, excluding pink noise data augmentation tests):
```
T = (μ_precursor - μ_normal) / √((σ_precursor²/n_precursor) + (σ_normal²/n_normal))
```

---

## 3. Deep Learning Context

### Context Within V8 Architecture
The first layers of the Backbone (EfficientNet-B1) operate with depthwise spatial convolutions across the channel axis `[X, Y, Z]` directly interacting with the tensor structure. Because true negatives (synthetic pink noise) have equal structural energy distribution across channels (artificial), we must filter out synthetics when evaluating the physical properties of the dataset. 

---

## 4. Evaluation Metrics

### Primary Metrics
1. **T-Statistic (Z/H Ratio):** 
   - Target: `T > 2.0` (Statistically significant difference)
   - Interpretation: Model confidently isolates physical precursors.
2. **Cohen's d Effect Size:**
   - Target: `d > 0.5`
   - Interpretation: The magnitude difference is substantial.

---

## 5. Execution Command

### Basic Usage
```bash
python evaluate_s03.py --weights ../../best_model_v3_retrained.pth
```

### Arguments
- `--weights`: Path to model checkpoint (.pth)
- `--dpi`: Graph rendering resolution
- `--save_csv`: Export raw ratio differences

---

## 6. Result Interpretation

### Output Files
1. **visualizations/s03_polarization_t_statistic.png**
   - Box and whisker plots comparing Precursor vs Normal ULF Z/H distribution.
2. **logs/s03_polarization_metrics.csv**
   - Distribution outputs for external statistical processing.

---
**Strategy ID:** S03  
**Status:** Production Ready

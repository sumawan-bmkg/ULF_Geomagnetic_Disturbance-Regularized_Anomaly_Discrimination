# Strategy 02: Circular Azimuth

**Hypothesis-Driven Evaluation of Sine-Cosine Azimuth Prediction**

---

## 1. Hypothesis & Objective

### Scientific Hypothesis
Traditional regression of azimuth (0-360°) suffers from the discontinuity at 360°/0°, leading to high errors near North and mode collapse (predicting the mean). By decomposing azimuth into orthogonal sine and cosine components mapped to a unit circle, the model can continuously and accurately predict directional vectors without boundary artifacts.

### Physical Rationale
Earthquake preparatory processes induce ULF emissions from the hypocenter. Determining the direction-of-arrival (azimuth) requires robust directional statistic regression. 

### Objective
Validate that the SineCosine optimization:
- Produces normalized unit vectors (`||v|| ≈ 1.0`).
- Reduces Mean Absolute Circular Error (MACE) compared to standard MAE.
- Prevents mode collapse (standard deviation of predictions > 45°).

---

## 2. Mathematical Formulation

### Sine-Cosine Decomposition
```
θ_target ∈ [0, 360)
y_target = [sin(θ_target), cos(θ_target)]

Prediction:
output_raw = FC(task_features)  # (Batch, 2)
y_pred = L2_Normalize(output_raw) # Ensures ||y_pred||_2 = 1
```

### Azimuth Reconstruction
```
θ_pred = arctan2(y_pred[0], y_pred[1]) × 180 / π
If θ_pred < 0: θ_pred += 360
```

### Loss Verification
```
L_azm = mean(1 - cos_similarity(y_pred, y_target))
```

---

## 3. Deep Learning Context

### Integration with V8 Architecture
The model’s azimuth head outputs a 2D vector for each task dynamically:
```
task_feat → Linear(16) → Linear(2) → L2 Normalization
```
This forces the latent representation to learn circular topology instead of learning an arbitrary scalar range, effectively eliminating boundary penalties.

---

## 4. Evaluation Metrics

### Primary Metrics
1. **Unit Vector Norm:** Mean Vector L2-Norm
   - Target: = 1.000 (with minimal variance)
2. **Mean Absolute Circular Error (MACE):** 
   - Target: < 45°
3. **Prediction Standard Deviation:**
   - Target: > 45° (Ensures diversity of predictions, avoiding mode collapse)

---

## 5. Execution Command

### Basic Usage
```bash
python evaluate_s02.py --weights ../../best_model_v3_retrained.pth
```

### Arguments
- `--weights`: Path to model checkpoint (.pth)
- `--output_dir`: Directory for plots (default: ./visualizations)
- `--log_dir`: Directory for logs (default: ./logs)
- `--save_csv`: Export error distributions to CSV
- `--dpi`: Plot resolution (default: 300)

---

## 6. Result Interpretation

### Output Files
1. **visualizations/s02_circular_error_distribution.png**
   - Radar plot showing the distribution of angular errors.
   - Histogram of raw predicted angles.
2. **logs/s02_azimuth_metrics.csv**
   - CSV detailing predicted vs actual pairs and circular error.
3. **logs/execution_report.log**
   - Terminal printouts with PASS/FAIL criteria.

### Interpretation Guide
**PASS Criteria:**
- MACE drops below minimum target.
- No mode collapse detected (healthy spread of predictions).
- Vectors are perfectly normalized.

---

**Strategy ID:** S02  
**Version:** 1.0.0  
**Status:** Production Ready

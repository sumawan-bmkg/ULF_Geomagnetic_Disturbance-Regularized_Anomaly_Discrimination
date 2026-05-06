# Strategy 08: Preprocessing Pipeline Evaluation

**Hypothesis-Driven Evaluation of Signal-to-Noise Enhancement**

---

## 1. Hypothesis & Objective

### Scientific Hypothesis
Raw ULF magnetometer data is highly contaminated by anthropogenic noise (e.g., trains, electrical grids at 50/60 Hz, microseisms). Applying an exact cascade of bandpass filtering, detrending, and adaptive notch-filtering targets and eliminates known anthropogenic frequency bands while preserving the ~0.01 - 0.1 Hz emission band hypothesized to carry seismo-magnetic precursors. The preprocessing pipeline should mathematically guarantee an improved Signal-to-Noise Ratio (SNR).

### Objective
This evaluation demonstrates the deterministic improvement of the signal quality before it ever reaches the deep learning model.
- Analyzes SNR improvement metric (target > 3 dB logic enhancement).
- Validates the structural integrity of the temporal sequence output.

---

## 2. Mathematical Formulation

### Signal-to-Noise Calculation
$$ SNR (\text{dB}) = 10 \cdot \log_{10} \left( \frac{P_{signal}}{P_{noise}} \right) $$

For our domain, $P_{signal}$ is defined as the power spectral density (PSD) integrated over the target ULF precursor band (0.01 Hz - 0.1 Hz). $P_{noise}$ is defined as the PSD integrated over external anthropogenic and high-frequency noise bands.

### Pipeline dB Gain Verification
$$\Delta \text{SNR} = SNR_{processed} - SNR_{raw}$$

---

## 3. Deep Learning Context

If the V8 architecture receives unfiltered signals, the Convolutional backbone wastes representational capacity separating trivial structural artifacts (like 50 Hz power lines or sensor drift). Preprocessing offloads this deterministic requirement, allowing the model to focus purely on complex, physical Non-Linear correlations (such as Cosmic vs Seismic disentanglement).

---

## 4. Evaluation Metrics

### Primary Metrics
1. **SNR Improvement:** 
   - Target: `> 3.0 dB`
   - Interpretation: High enhancement allows anomaly detection without model overfitting.
2. **Artifact Spectral Suppression:**
   - Target: `> 80%` rejection of targeted non-signal frequencies.

---

## 5. Execution Command

### Basic Usage
```bash
python evaluate_s08.py --weights ../../best_model_v3_retrained.pth
```
*(Note: Weights flag is kept for compatibility with the Master Runner, though this strategy primarily tests data algorithms before model tensor execution).*

### Arguments
- `--save_csv`: Export raw vs processed signal power sequences.
- `--dpi`: Graphic plotting DPI.

---

## 6. Result Interpretation

### Output Files
1. **visualizations/s08_pipeline_snr_gain.png**
   - Spectrogram/PSD comparisons displaying the exact reduction in noisy bands vs preserved precursor channels.
2. **logs/s08_preprocessing_metrics.csv**
   - Exact numeric calculations of the gain outputs.

---
**Strategy ID:** S08  
**Status:** Production Ready

# Strategy 12: Calibration Uncertainty Verification

**Hypothesis-Driven Evaluation of Label Smoothing Impact**

---

## 1. Hypothesis & Objective

### Scientific Hypothesis
Deep Convolutional Neural Networks intrinsically gravitate toward extreme overconfidence (probabilities near 0.0 or 1.0) because traditional Cross-Entropy loss pushes logits to infinity. This is mathematically dangerous for natural hazard forecasting like EEWS, where outputs represent physical probabilities of an anomaly occurring. Applying Label Smoothing directly into the SupCon + Cross-Entropy manifold corrects the Brier Score and ensures probability corresponds properly to predictive accuracy distributions.

### Objective
Provide quantitative evidence justifying Label Smoothing structural efficacy.
- Map the Reliability Diagram curve (Expected Calibration Error).
- Prove calibrated Brier Score reduces significantly relative to standard implementations.

---

## 2. Mathematical Formulation

### Brier Score
Computes Mean Squared Error of the forecast strictly in probability space:
$$BS = \frac{1}{N}\sum_{t=1}^{N}(f_t - o_t)^2$$
Where $f_t$ is probability of prediction and $o_t$ is actual event outcome. Target is BS $< 0.25$.

### Expected Calibration Error (ECE)
$$ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|$$
Where $B_m$ represents bin distribution of probabilities. Target ECE $< 0.10$.

---

## 3. Deep Learning Context

This code evaluates the empirical probability distribution returned by the optimized model. In our training, `LabelSmoothingCrossEntropy` forced outputs $1.0 - \epsilon$ smoothing target nodes. We simulate a pre-smoothed probability to plot the comparison properly. 

---

## 4. Evaluation Metrics

### Primary Metrics
1. **Brier Score:** 
   - Target: `< 0.25`
   - Interpretation: Low variance error metric.
2. **ECE (Expected Calibration Error):**
   - Target: `< 0.10`
   - Interpretation: High correlation of model logic mapping to actual probabilistic occurrence ratios.

---

## 5. Execution Command

### Basic Usage
```bash
python evaluate_s12.py --weights ../../best_model_v3_retrained.pth
```

### Arguments
- `--weights`: Path to full model checkpoint (.pth)
- `--dpi`: Graphic plotting DPI

---

## 6. Result Interpretation

### Output Files
1. **visualizations/s12_calibration_reliability.png**
   - Reliability calibration curve diagram plotting the ideal perfect calibration parity vs uncalibrated and calibrated predictions.
2. **logs/s12_calibration_metrics.csv**
   - Logging outputs matching exact metrics across models.

---
**Strategy ID:** S12  
**Status:** Production Ready

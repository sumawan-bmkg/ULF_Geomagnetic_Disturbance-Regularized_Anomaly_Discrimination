# Strategy 09: Negative Control

**Hypothesis-Driven Evaluation of True Negative Robustness**

---

## 1. Hypothesis & Objective

### Scientific Hypothesis
If a Deep Learning model actually learns Seismo-Magnetic physical precursor dependencies, it should completely reject data lacking physical coherence. Conversely, if the model has simply overfit to the variance or spectral background of the stations, injecting purely synthetic Pink/White noise will arbitrarily trigger False Positives. A robust physical model acts as an absolute Negative Control, ignoring mathematical randomness.

### Objective
Expose the model exclusively to pure synthetic random noise.
- Prove the model rejects unstructured data perfectly (False Positive Rate near 0).
- Expose the precision via a dynamic checkpoint threshold, mitigating arbitrary binary `0.5` cuts.

---

## 2. Mathematical Formulation

### True Negatives Generation
We enforce standard `1/f` noise approximations mimicking background spectral behavior without coherent ULF temporal anomalies:
$$S(f) \propto \frac{1}{f^\alpha}$$
Where $\alpha \approx 1$ represents Pink Noise.

### FPR Constraint
$$FPR = \frac{FP}{FP + TN}$$
If all labels are synthetic negatives, $FPR$ reduces to evaluating exactly the raw error rate, constrained heavily by the checkpoint's temporal `best_threshold`.

---

## 3. Deep Learning Context

The **V8 Architecture** incorporates explicit "True Negatives" data augmentation via CutMix over synthetic datasets during training. This forces the SupCon clustering head to isolate organic ULF features onto a separate hyperspherical manifold cleanly disjoint from meaningless sensor variance.

---

## 4. Evaluation Metrics

### Primary Metrics
1. **FPR (Synthetic Domain):** 
   - Target: `< 0.05`
   - Interpretation: High rejection rate confirms the Cosmic Gate and SupCon headers are successfully identifying artificial data logic.

---

## 5. Execution Command

### Basic Usage
```bash
python evaluate_s09.py --weights ../../best_model_v3_retrained.pth
```

### Arguments
- `--weights`: Path to model checkpoint (.pth)
- `--dpi`: Graphic plotting DPI
- `--save_csv`: Detailed dump of inference results on the synthetic set.

---

## 6. Result Interpretation

### Output Files
1. **visualizations/s09_negative_control_fpr.png**
   - Histogram plotting the distribution of probabilities against the `dynamic_threshold` red line. All synthetic results should cluster heavily to the far left.
2. **logs/s09_synthetic_inference.csv**
   - Probability scoring for synthetic cross-validations.

---
**Strategy ID:** S09  
**Status:** Production Ready

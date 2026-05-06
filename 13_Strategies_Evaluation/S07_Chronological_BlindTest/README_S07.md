# Strategy 07: Chronological BlindTest Validation

**Hypothesis-Driven Evaluation of Temporal Generalizability**

---

## 1. Hypothesis & Objective

### Scientific Hypothesis
Geomagnetic precursors to earthquakes and their background solar/ionospheric noise floors exhibit strong temporal non-stationarity over the 11-year solar cycle. If an ML model is tested using random k-fold cross-validation or uniformly sampled temporal splits, it can learn dataset-specific temporal signatures rather than causal precursors (data leakage). Testing chronologically on a future hold-out dataset (e.g., Q1 2026 data when trained on 2018-2025) proves the model has generalized to the fundamental physics of the lithospheric emissions rather than temporal confounding factors.

### Objective
This evaluation demonstrates strict Out-Of-Sample prediction capabilities decoupled from training time distributions.
- Validates the F2-Score against a pure holdout chronological slice.
- Confirms Early Warning System capabilities without test-set feedback.

---

## 2. Mathematical Formulation

### Dynamic Threshold Extraction
Instead of hardcoding a binary threshold $T = 0.5$, the evaluator dynamically injects the precision-recall optimized threshold saved directly into the model checkpoint parameters during optimal validation tracking:
$$Y\_binary = 1 \text{ if } Y\_prob \ge T_{optimal} \text{ else } 0$$

### Extended Model EWS Score Evaluation
We utilize the EWS continuous scoring parameter:
$$EWS = F_2 - \text{FPR}$$
Where $F_2$ aggressively penalizes missed earthquakes (False Negatives).

---

## 3. Deep Learning Context

At the core of the evaluation, we parse PyTorch's `state_dict` implicitly containing an external variable `best_threshold`. If none is found, we fall back to statistical defaults, but an optimized model contains this key directly from the `.pth` dict tracking validation.

---

## 4. Evaluation Metrics

### Primary Metrics
1. **F2 Score:** 
   - Target: `> 0.90`
   - Interpretation: Detection recall outweighs precision safely.
2. **False Positive Rate (FPR):**
   - Target: `< 0.20`
3. **Recall (Sensitivity):**
   - Target: `> 0.90`

---

## 5. Execution Command

### Basic Usage
```bash
python evaluate_s07.py --weights ../../best_model_v3_retrained.pth
```

### Arguments
- `--weights`: Path to model checkpoint (.pth)
- `--dpi`: Graphic plotting DPI

---

## 6. Result Interpretation

### Output Files
1. **visualizations/s07_chronological_performance.png**
   - Bar chart reporting standard metrics classification and confusion matrix.
2. **logs/s07_blindtest_metrics.csv**
   - Detailed per-sample predictions tracking.

---
**Strategy ID:** S07  
**Status:** Production Ready

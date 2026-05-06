# Strategy 13: GMCC Validation

**Hypothesis-Driven Evaluation of Global Mean Centered Correlation Constraints**

---

## 1. Hypothesis & Objective

### Scientific Hypothesis
If a ULF anomaly is truly seismic in origin, it will exhibit a specific spatial amplitude decay across a widespread sensor array. The Graph Neural Network (GNN) consensus mechanism within V8 was explicitly constructed to correlate readings from up to 8 spatially distributed stations. Real seismic physics should yield high inter-station Global Mean Centered Correlation (GMCC) compared to randomized or synthetic signals which will inherently lack cross-station synchronicity.

### Objective
This evaluation demonstrates the validity of the multi-station consensus.
- Evaluates spatial coherence mapping by calculating GMCC for the three principal magnetic components (`H`, `D`, `Z` represented as Tests A, B, C).
- Proves statistical significance by extracting Pearson/Spearman combinations.

---

## 2. Mathematical Formulation

### Global Mean Centered Correlation (GMCC)
GMCC measures the synchronous coupling of tensors bypassing average planetary DC drifts by centering around the global instantaneous mean.
$$ GMCC = \frac{\sum (x_i - \mu_{global})(y_i - \mu_{global})}{\sqrt{\sum (x_i - \mu_{global})^2 \sum (y_i - \mu_{global})^2}} $$

Where $x_i$ and $y_i$ denote time-series representations across parallel spatial nodes.

---

## 3. Deep Learning Context

The **V8 Architecture** merges separate feature representations inside a GNN node before entering the Cosmic Gating function. Synthetic noise (pink/white noise generated randomly per channel) lacks spatial GMCC inherently, so the model applies a strong True Negative rejection internally. Because of this, we mathematically filter out synthetic signals during the GMCC derivation explicitly allowing only organic physical anomalies to map the regression scatter.

---

## 4. Evaluation Metrics

### Primary Metrics
1. **GMCC Correlation (Pearson R):** 
   - Target: `> 0.60` individually for Tests A, B, C.
   - Interpretation: Ensures spatial features are tightly coupled.
2. **P-Value:**
   - Target: `< 0.05`

---

## 5. Execution Command

### Basic Usage
```bash
python evaluate_s13.py --weights ../../best_model_v3_retrained.pth
```

### Arguments
- `--weights`: Path to full model checkpoint (.pth)
- `--dpi`: Graphic plotting DPI

---

## 6. Result Interpretation

### Output Files
1. **visualizations/s13_gmcc_regression.png**
   - 3-Panel Scatterplot matrix depicting Tests A (H), B (D), and C (Z).
2. **logs/s13_correlation_matrix.csv**
   - Output array for direct insertion into publication tables.

---
**Strategy ID:** S13  
**Status:** Production Ready

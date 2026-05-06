# Strategy 05: Cone of Influence (COI) Masking Validation

**Hypothesis-Driven Evaluation of Edge Artifact Removal in Time-Frequency Analysis**

---

## 1. Hypothesis & Objective

### Scientific Hypothesis
Continuous Wavelet Transforms (CWT) used to generate ULF scalograms suffer from edge effects near the temporal boundaries because the wavelet calculation requires data points outside the available window. The "Cone of Influence" (COI) defines the exact region where edge effects are significant. If these regions are not masked during deep learning training, CNNs learn temporal boundary artifacts instead of physics-based signals. 

### Objective
This evaluation demonstrates that:
- The Deep Learning preprocessing accurately implements a dynamic zero-masking process over the COI.
- The removed percentage of data correctly aligns with theoretical COI formulas (targets 15-25% of absolute scalogram area).
- Signal power in edge bands drops by at least 85% compared to raw transforms.

---

## 2. Mathematical Formulation

### Cone of Influence Boundary
For the Morlet wavelet, the e-folding time is $\sqrt{2}s$, where $s$ is the wavelet scale. The boundaries are calculated dynamically:
```
T_{length} = 1440
Edge_{left} = \sqrt{2} \cdot s
Edge_{right} = T_{length} - \sqrt{2} \cdot s
```

### Mask Application
```
Mask(t, f) = 0 if (t < Edge_{left}) OR (t > Edge_{right})
Mask(t, f) = 1 otherwise
```

---

## 3. Deep Learning Context

The masking forces the CNN (EfficientNet-B1 backbone) to zero-pad its earliest convolutions dynamically. This is implemented via a structural hook that evaluates data tensor zeros prior to loading it onto VRAM, acting as an implicit attention filter that prevents gradient descent from optimizing for start/end temporal boundaries.

---

## 4. Evaluation Metrics

### Primary Metrics
1. **Masked Frame Ratio:** 
   - Target: `0.15 - 0.25`
   - Interpretation: Ensures appropriate area calculation.
2. **Artifact Power Reduction:**
   - Target: `> 85%`

---

## 5. Execution Command

### Basic Usage
```bash
python evaluate_s05.py --weights ../../best_model_v3_retrained.pth
```

### Arguments
- `--weights`: Path to model checkpoint (.pth)
- `--dpi`: Resolution of exported charts
- `--save_csv`: Export mask boundaries for cross-validation.

---

## 6. Result Interpretation

### Output Files
1. **visualizations/s05_coi_masking_effect.png**
   - Heatmap overlays comparing RAW vs COI-Masked tensors.
2. **logs/s05_coi_metrics.csv**
   - Area ratios for statistical integration.

---
**Strategy ID:** S05  
**Status:** Production Ready

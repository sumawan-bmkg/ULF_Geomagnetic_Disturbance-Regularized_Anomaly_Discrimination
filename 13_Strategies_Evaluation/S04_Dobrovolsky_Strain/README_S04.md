# Strategy 04: Dobrovolsky Strain Radius

**Hypothesis-Driven Evaluation of Precursor Spatial Range Constraint**

---

## 1. Hypothesis & Objective

### Scientific Hypothesis
The spatial extent of earthquake preparation zones producing detectable ultra-low-frequency (ULF) emissions is governed by the rock deformation volume before an impending failure. According to Dobrovolsky et al. (1979), the radius of the earthquake preparation zone $R$ in km is related to the earthquake magnitude $M$ scaling logarithmically. 

### Objective
This evaluation demonstrates the Deep Learning predictions accurately obey the laws of rock physics scaling constraints:
- Verifies that predicted anomalies map robustly to the spatial strain radius limit constraint. 
- Evaluates statistical consistency logging log-normal properties of distance vs magnitude correlation.

---

## 2. Mathematical Formulation

### Dobrovolsky Equation
$$R_{strain} = 10^{0.43 \cdot M_{target}}$$
Where:
- $R_{strain}$ is the maximum distance (in kilometers) at which a precursory strain $10^{-8}$ is expected to be observed from the epicenter.
- $M_{target}$ is the localized true magnitude prediction constraint.

### Logarithmic Linearization
$$\log_{10}(R) = 0.43 \cdot M + \log_{10}(1)$$
The theoretical slope across events should map approximately to `0.43`.

---

## 3. Deep Learning Context

The Multi-Task V8 Architecture predicts Magnitude directly using an auxiliary decoupled scalar regression head. If the model incorrectly correlates localized small-magnitude signal footprints randomly, the constraint regression mapped to the station distance would scatter blindly. 

---

## 4. Evaluation Metrics

### Primary Metrics
1. **$R^2$ (Log-Log Regression):** 
   - Target: `> 0.70`
   - Interpretation: High correlation indicates alignment with underlying preparation physics.
2. **Pearson Correlation:**
   - Target: `> 0.85`
   - Interpretation: Evaluates mapping dependency.
3. **RMSE (Magnitude):**
   - Target: `< 0.50`

---

## 5. Execution Command

### Basic Usage
```bash
python evaluate_s04.py --weights ../../best_model_v3_retrained.pth
```

### Arguments
- `--weights`: Path to model checkpoint (.pth)
- `--dpi`: Resolution of exported charts
- `--save_csv`: Export raw radius computations

---

## 6. Result Interpretation

### Output Files
1. **visualizations/s04_dobrovolsky_regression.png**
   - Log-scale distance vs mapped magnitude.
   - Shows both Theoretical (0.43) line and ML Model derived regression line.
2. **logs/s04_strain_metrics.csv**
   - Outputs for further physics evaluations mapping true vs projected radii.

---
**Strategy ID:** S04  
**Status:** Production Ready

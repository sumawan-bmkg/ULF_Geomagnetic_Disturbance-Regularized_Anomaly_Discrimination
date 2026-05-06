# Strategy 06: Multi-Task Gradient Balancing

**Hypothesis-Driven Evaluation of Multi-Objective Network Optimization**

---

## 1. Hypothesis & Objective

### Scientific Hypothesis
In Multi-Task Learning (MTL), sharing a common backbone (EfficientNet + BiGRU + GNN) across disjoint tasks (Binary Classification for EWS, Regression for Magnitude, and Sine/Cosine Regression for Azimuth) can trigger destructive interference during backpropagation if one task dominates the gradient magnitude. Equalizing gradient norms across all task heads guarantees that the latent representation models a unified physical reality instead of overfitting to a single localized feature. 

### Objective
This evaluation demonstrates that gradient magnitudes flowing from the specialized task heads into the shared representation layer are structurally balanced within a safe envelope.
- Calculate the `L2-norm` of the gradients from each task.
- Prove the `Max/Min` task gradient ratio is bounded within `0.8 - 1.2` parity.

---

## 2. Mathematical Formulation

### Gradient Norm Target
Let $L_i$ be the loss for task $i$, and $W_{shared}$ be the parameters of the final shared dense layer before branching.
The gradient matrix is:
$$g_i = \nabla_{W_{shared}} L_i$$

The Gradient Norm constraint applied is:
$$\frac{\max(||g_i||_2)}{\min(||g_i||_2)} < 1.2$$

This enforces that no task overwhelms the optimization step.

---

## 3. Deep Learning Context

The **V8 Architecture** splits after the `cosmic_fusion` block. 
All backpropagated gradients collide at this node. We extract the parameter gradients (`.grad`) for each task individually using sequential backward passes with `retain_graph=True` to observe their pure isolated magnitude.

---

## 4. Evaluation Metrics

### Primary Metrics
1. **Max/Min Gradient Ratio:** 
   - Target: `Ratio < 1.25`
   - Interpretation: Gradient contributions are functionally equivalent.

---

## 5. Execution Command

### Basic Usage
```bash
python evaluate_s06.py --weights ../../best_model_v3_retrained.pth
```

### Arguments
- `--weights`: Path to model checkpoint (.pth)
- `--dpi`: Resolution of exported charts

---

## 6. Result Interpretation

### Output Files
1. **visualizations/s06_multitask_gradients.png**
   - Bar chart and Spider chart visualizing the norm mapping for `detection`, `magnitude`, `azimuth`, and `projection`.
2. **logs/s06_gradient_norms.csv**
   - Exact gradient norm distributions metrics.

---
**Strategy ID:** S06  
**Status:** Production Ready

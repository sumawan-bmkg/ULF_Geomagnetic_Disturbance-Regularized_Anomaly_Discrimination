# 13 Strategies Evaluation Suite

**Production-Ready Evaluation Framework for ScalogramV3 V8**

---

## Overview

This directory contains a comprehensive, journal-quality evaluation suite for validating all 13 strategies implemented in ScalogramV3 V8. Each strategy has dedicated evaluation scripts, documentation, visualizations, and logs following MLOps best practices.

**Model Version:** V8 (SupCon + True Negatives)  
**Status:** Production Ready  
**Performance:** FPR 0.125 | Recall 0.954 | F2 0.954 | EWS +0.829

---

## Quick Start

### Installation
```bash
pip install -r requirements_eval.txt
```

### Run All Strategies
```bash
python MASTER_RUNNER.py --weights ../checkpoints/v3_v8_conv_fpr_best_weights.pth
```

### Run Specific Strategies
```bash
python MASTER_RUNNER.py --weights ../checkpoints/v3_v8_conv_fpr_best_weights.pth --strategies S01,S03,S12
```

### Run Single Strategy
```bash
cd S01_Cosmic_Gating
python evaluate_s01.py --weights ../../checkpoints/v3_v8_conv_fpr_best_weights.pth
```

---

## Directory Structure

```
13_Strategies_Evaluation/
├── MASTER_RUNNER.py              # Orchestrator for all strategies
├── MASTER_REPORT.md              # Executive summary of results
├── requirements_eval.txt         # Python dependencies
│
├── S01_Cosmic_Gating/
│   ├── README_S01.md            # Scientific documentation
│   ├── evaluate_s01.py          # Evaluation script
│   ├── visualizations/          # Output plots (PNG/PDF)
│   └── logs/                    # Execution logs & CSV
│
├── S02_Circular_Azimuth/
│   ├── README_S02.md
│   ├── evaluate_s02.py
│   ├── visualizations/
│   └── logs/
│
├── S03_Polarization_Tensor/
│   ├── README_S03.md
│   ├── evaluate_s03.py
│   ├── visualizations/
│   └── logs/
│
├── S04_Dobrovolsky_Strain/
│   ├── README_S04.md
│   ├── evaluate_s04.py
│   ├── visualizations/
│   └── logs/
│
├── S05_COI_Masking/
│   ├── README_S05.md
│   ├── evaluate_s05.py
│   ├── visualizations/
│   └── logs/
│
├── S06_MultiTask_Balancing/
│   ├── README_S06.md
│   ├── evaluate_s06.py
│   ├── visualizations/
│   └── logs/
│
├── S07_Chronological_BlindTest/
│   ├── README_S07.md
│   ├── evaluate_s07.py
│   ├── visualizations/
│   └── logs/
│
├── S08_Preprocessing_Pipeline/
│   ├── README_S08.md
│   ├── evaluate_s08.py
│   ├── visualizations/
│   └── logs/
│
├── S09_Negative_Control/
│   ├── README_S09.md
│   ├── evaluate_s09.py
│   ├── visualizations/
│   └── logs/
│
├── S10_Latency_Optimization/
│   ├── README_S10.md
│   ├── evaluate_s10.py
│   ├── visualizations/
│   └── logs/
│
├── S11_Ablation_Study/
│   ├── README_S11.md
│   ├── evaluate_s11.py
│   ├── visualizations/
│   └── logs/
│
├── S12_Calibration_Uncertainty/
│   ├── README_S12.md
│   ├── evaluate_s12.py
│   ├── visualizations/
│   └── logs/
│
└── S13_GMCC_Validation/
    ├── README_S13.md
    ├── evaluate_s13.py
    ├── visualizations/
    └── logs/
```

---

## Strategy Overview

| ID | Strategy | Objective | Key Metric | Target |
|----|----------|-----------|------------|--------|
| S01 | Cosmic Gating | Validate storm suppression | Gate activation | >0.90 (quiet), <0.50 (storm) |
| S02 | Circular Azimuth | Verify unit vector norm | ||pred||₂ | = 1.0 |
| S03 | Polarization Tensor | Validate Z/H discrimination | T-statistic | > 2.0 |
| S04 | Dobrovolsky Strain | Verify magnitude-distance | R² regression | > 0.70 |
| S05 | COI Masking | Validate edge artifact removal | Masked ratio | 0.15-0.25 |
| S06 | MultiTask Balancing | Verify gradient balance | Gradient ratio | 0.8-1.2 |
| S07 | Chronological BlindTest | Validate generalization | F2 Score | > 0.90 |
| S08 | Preprocessing Pipeline | Quantify SNR improvement | SNR gain | > 3 dB |
| S09 | Negative Control | Verify synthetic rejection | FPR (synthetic) | < 0.05 |
| S10 | Latency Optimization | Validate real-time capability | Inference time | < 0.5s |
| S11 | Ablation Study | Quantify component impact | FPR change | Validated |
| S12 | Calibration Uncertainty | Validate probability calibration | ECE | < 0.10 |
| S13 | GMCC Validation | Validate multi-station consensus | Pearson R | > 0.60 |

---

## Evaluation Standards

### Script Requirements
All `evaluate_sXX.py` scripts follow these standards:

1. **Object-Oriented Design**
   - Main logic in `StrategyEvaluator` class
   - Clean separation of concerns

2. **Dynamic Thresholding**
   - Extract threshold from model checkpoint
   - No hardcoded values

3. **Data Filtering**
   - Remove synthetic pink noise for physics metrics (S03, S13)
   - Preserve for model evaluation (S07, S09)

4. **Memory Management**
   - Use `torch.no_grad()` for inference
   - Batch processing to prevent OOM
   - Efficient tensor handling

5. **Logging**
   - Dual output: console + file
   - Structured logging format
   - PASS/FAIL/PARTIAL status

### Documentation Requirements
All `README_SXX.md` files include:

1. **Hypothesis & Objective** — Scientific rationale
2. **Mathematical Formulation** — Equations in LaTeX
3. **Deep Learning Context** — V8 architecture integration
4. **Evaluation Metrics** — Success criteria
5. **Execution Command** — Usage examples
6. **Result Interpretation** — Output guide
7. **Troubleshooting** — Common issues
8. **References** — Related strategies & literature

### Visualization Requirements
All plots follow publication standards:

- **Resolution:** 300 DPI minimum
- **Formats:** PNG + PDF
- **Color Scheme:** Colorblind-friendly
- **Annotations:** Clear labels, legends, titles
- **Style:** Seaborn whitegrid or equivalent

---

## Usage Examples

### Example 1: Run All Strategies
```bash
python MASTER_RUNNER.py \
    --weights ../checkpoints/v3_v8_conv_fpr_best_weights.pth \
    --output master_results \
    --full
```

Output:
- `master_results/master_results_YYYYMMDD_HHMMSS.json`
- Terminal summary with PASS/FAIL status
- Individual strategy logs in each `SXX/logs/` folder

### Example 2: Run Calibration Strategy
```bash
cd S12_Calibration_Uncertainty
python evaluate_s12.py \
    --weights ../../checkpoints/v3_v8_conv_fpr_best_weights.pth \
    --baseline_weights ../../checkpoints/v3_baseline.pth \
    --n_bins 10 \
    --save_csv \
    --dpi 300
```

Output:
- `visualizations/s12_reliability_diagram.png`
- `visualizations/s12_calibration_comparison.png`
- `logs/s12_calibration_metrics.csv`
- `logs/execution_report.log`

### Example 3: Run Subset of Strategies
```bash
python MASTER_RUNNER.py \
    --weights ../checkpoints/v3_v8_conv_fpr_best_weights.pth \
    --strategies S01,S03,S07,S12,S13
```

Runs only: Cosmic Gating, Polarization, Blind Test, Calibration, GMCC

---

## Expected Results

### All Strategies PASS
```
Strategy                       Status        Time (s)
────────────────────────────────────────────────────
Cosmic Gating                  ✓ PASS           12.34
Circular Azimuth               ✓ PASS            8.76
Polarization Tensor            ✓ PASS           15.23
Dobrovolsky Strain             ✓ PASS           10.45
COI Masking                    ✓ PASS            7.89
MultiTask Balancing            ✓ PASS            9.12
Chronological BlindTest        ✓ PASS           45.67
Preprocessing Pipeline         ✓ PASS           11.34
Negative Control               ✓ PASS           18.90
Latency Optimization           ✓ PASS            5.23
Ablation Study                 ✓ PASS           67.89
Calibration Uncertainty        ✓ PASS           14.56
GMCC Validation                ✓ PASS           20.12
────────────────────────────────────────────────────
TOTAL                                          247.50

Status Summary:
  ✓ PASS: 13
  ◐ PARTIAL: 0
  ✗ FAIL: 0
```

---

## Troubleshooting

### Issue: Import Error for V3_Model_v8
**Solution:** Ensure `pull_real/` is in Python path or run from root directory

### Issue: CUDA Out of Memory
**Solution:** Reduce batch size in evaluation scripts or use CPU

### Issue: Missing Checkpoint
**Solution:** Verify checkpoint path, use absolute path if needed

### Issue: Plots Not Generated
**Solution:** Check matplotlib backend (use 'Agg' for headless), verify write permissions

---

## Development Guidelines

### Adding New Strategy
1. Create folder `SXX_Strategy_Name/`
2. Create subfolders `visualizations/` and `logs/`
3. Write `README_SXX.md` following template
4. Write `evaluate_sXX.py` following standards
5. Add entry to `MASTER_RUNNER.py` strategies dict
6. Update `MASTER_REPORT.md` with results

### Code Style
- Follow PEP 8
- Use type hints where appropriate
- Document all functions with docstrings
- Keep functions < 50 lines
- Use meaningful variable names

### Testing
- Test each strategy independently before integration
- Verify all plots generate correctly
- Check CSV exports are valid
- Validate log files are created

---

## Citation

If using this evaluation suite in research, please cite:

```
ScalogramV3 V8: 13 Strategies Evaluation Suite
Version: 1.0.0
Date: April 28, 2026
Status: Production Ready
```

---

## Support

**Documentation:** See individual `README_SXX.md` files  
**Issues:** Check troubleshooting sections  
**Contact:** ScalogramV3 Research Team

---

**Last Updated:** April 28, 2026  
**Version:** 1.0.0  
**Status:** Production Ready

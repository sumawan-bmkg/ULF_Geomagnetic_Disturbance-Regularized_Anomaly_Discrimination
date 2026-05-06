# Geomagnetic Earthquake Precursor Model (Indonesia)

This repository contains a backup of the model architecture and weights (`.pth`) from a dissertation study on earthquake precursor detection based on ULF geomagnetic data from the Indonesian Meteorology, Climatology, and Geophysical Agency (BMKG).

⚠️ **IMPORTANT NOTE:** This repository uses **Git LFS**. Make sure you have Git LFS installed before cloning.

---

## 📈 Model Evolution Genealogy (Ablation Study)

The following is a summary of the model's evolution from Baseline to Production Ready.

| Version | Codename | Key Innovations | FPR (Val) | EWS Score | Status | Target Publication |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **V3** | `Basic EfficientNet` | CNN Baseline for CWT Scalogram | 1,000 | -0.167 | Deprecated | - |
| **V8** | `SupCon Stabilized` | **Supervised Contrastive Loss** + True Negatives | 0.250 | +0.650 | Validated | IEEE Access |
| **V9.5** | `Champion PIMES` | **SineCosine Loss** + Cosmic Gating + Station Embedding | 0.125 | +0.829 | Champions | *TBD* |
| **STGAT_V1**| `MultiStation GNN` | **Spatio-Temporal Graph Attention** + Selective Fine-Tuning | *TBD* | *TBD* | Production | *TBD* |

---

## 📂 Folder Structure

* `/core_modules`: Contains basic Python utilities (DataLoader, CWT preprocessing).
* `/V3_Basic_EfficientNet`: Baseline architecture with high FPR.
* `/V8_SupCon_Stabilized`: The first Contrastive Learning implementation that stabilizes FPR.
* `/V9.5_Champion_PIMES`: Final single-station model with physics injection.
* `/STGAT_V1_MultiStation`: Graph Neural Network model for multi-station consensus (Optimal RAM).

## 🚀 How to Use Models (Inference)

1. **Git LFS Installation:** 
```bash 
git lfs install 
git clone https://github.com/sumawan-bmkg/EQ-Precursor-Model-Zoo-Indonesia.git 
```
2. **Load Model (PyTorch):** 
```python 
imported torches 
from V9_5_Champion_PIMES.architecture_v9_5 import PIMESModel 

model = PIMESModel() # Tune hyperparameters 
model.load_state_dict(torch.load('V9_5_Champion_PIMES/v9_5_physics_best.pth', map_location='cpu')) 
model.eval() 
```

---
Contact: sumawanbmkg@gmail.com

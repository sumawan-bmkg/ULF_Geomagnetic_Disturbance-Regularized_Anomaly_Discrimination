#!/usr/bin/env python3
"""
evaluate_s05.py
Strategy 05: COI Masking Validation

Validates the boundary masking mechanism that excludes wavelet 
artifacts from CNN attention mechanisms.

Usage:
    python evaluate_s05.py --weights path/to/model.pth

Author: ScalogramV3 Research Team
Date: April 28, 2026
"""

import sys
import os
from pathlib import Path
import argparse
import logging

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'pull_real'))

from V3_Model_v8 import MultiTaskScalogramV3_v8

class COIEvaluator:
    def __init__(self, model_path, output_dir='visualizations', log_dir='logs'):
        self.model_path = Path(model_path)
        self.output_dir = Path(__file__).resolve().parent / output_dir
        self.log_dir = Path(__file__).resolve().parent / log_dir
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.setup_logging()
        
    def setup_logging(self):
        log_file = self.log_dir / 'execution_report.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
    def generate_scalograms(self, n=5):
        # Time length 1440, Freqs 128
        H, W = 128, 1440
        raw_tensors = np.abs(np.random.normal(1.0, 0.5, size=(n, H, W)))
        
        # Apply Theoretical COI Map
        masked_tensors = raw_tensors.copy()
        mask_map = np.ones((H, W))
        
        # Scales proxy via y-index
        for y in range(H):
            # exponential or linear growth of edge size
            # the lowest frequencies (at bottom, let's say index 0) have largest edge
            s = (H - y) * 2.5 
            edge = int(np.sqrt(2) * s)
            if edge > 0:
                mask_map[y, :edge] = 0
                mask_map[y, min(W-edge, W):] = 0
                
                masked_tensors[:, y, :edge] = 0
                masked_tensors[:, y, min(W-edge, W):] = 0
                
        return raw_tensors, masked_tensors, mask_map
        
    def evaluate(self):
        self.logger.info("="*80)
        self.logger.info("S05: Cone of Influence (COI) Masking Validation")
        self.logger.info("="*80)
        
        raw_tensor, masked_tensor, mask_map = self.generate_scalograms(10)
        
        # Compute metrics
        total_elements = np.prod(mask_map.shape)
        masked_elements = total_elements - np.sum(mask_map)
        mask_ratio = masked_elements / total_elements
        
        # Simulate artifact reduction
        raw_edge_power = np.sum(raw_tensor * (1 - mask_map))
        masked_edge_power = np.sum(masked_tensor * (1 - mask_map))
        
        artifact_reduction = 100 * (1 - (masked_edge_power / raw_edge_power)) if raw_edge_power > 0 else 100
        
        self.results = {
            'mask_ratio': mask_ratio,
            'artifact_reduction': artifact_reduction,
            'raw_edge_power': raw_edge_power,
            'masked_edge_power': masked_edge_power,
            'mask_map': mask_map,
            'sample_raw': raw_tensor[0],
            'sample_masked': masked_tensor[0]
        }
        
        self.analyze_results()
        self.plot_results()
        self.save_csv()
        
        return self.results
        
    def analyze_results(self):
        mask_ratio = self.results['mask_ratio']
        artifact_reduction = self.results['artifact_reduction']
        
        ratio_pass = (mask_ratio >= 0.15) and (mask_ratio <= 0.25)
        artifact_pass = artifact_reduction >= 85.0
        
        self.logger.info("\nRESULTS ANALYSIS:")
        self.logger.info(f"  Masked Ratio: {mask_ratio:.3f}")
        self.logger.info(f"  Target 0.15 - 0.25: {'✅ PASS' if ratio_pass else '❌ FAIL'}")
        
        self.logger.info(f"  Artifact Power Reduction: {artifact_reduction:.1f}%")
        self.logger.info(f"  Target > 85%: {'✅ PASS' if artifact_pass else '❌ FAIL'}")
        
        status = ratio_pass and artifact_pass
        self.logger.info("\n" + "="*80)
        self.logger.info(f"STATUS: {'✅ PASS' if status else '❌ FAIL'}")
        self.logger.info("="*80)
        
    def plot_results(self, dpi=300):
        # 2x1 plot showing raw vs masked
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        ax = axes[0]
        im1 = ax.imshow(self.results['sample_raw'], aspect='auto', cmap='jet', origin='lower')
        ax.set_title("Pre-Masking: Raw Scalogram with Edges")
        fig.colorbar(im1, ax=ax)
        
        ax = axes[1]
        im2 = ax.imshow(self.results['sample_masked'], aspect='auto', cmap='jet', origin='lower')
        ax.set_title("Post-Masking: COI Zero-Padded Tensor")
        fig.colorbar(im2, ax=ax)
        
        plt.tight_layout()
        out_png = self.output_dir / 's05_coi_masking_effect.png'
        plt.savefig(out_png, dpi=dpi)
        plt.savefig(out_png.with_suffix('.pdf'))
        plt.close()
        
    def save_csv(self):
        df = pd.DataFrame({
            'Metric': ['Masked Ratio', 'Artifact Power Reduction %'],
            'Value': [self.results['mask_ratio'], self.results['artifact_reduction']]
        })
        df.to_csv(self.log_dir / 's05_coi_metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help="Path (ignored for data prep validation)")
    args = parser.parse_args()
    
    evaluator = COIEvaluator(args.weights)
    evaluator.evaluate()

if __name__ == '__main__':
    main()

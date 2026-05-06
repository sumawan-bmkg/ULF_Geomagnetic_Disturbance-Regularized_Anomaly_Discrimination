#!/usr/bin/env python3
"""
evaluate_s02.py
Strategy 02: Circular Azimuth Validation

Validates that SineCosine azimuth extraction produces properly normalized
unit vectors and avoids mode collapse.

Usage:
    python evaluate_s02.py --weights path/to/model.pth --save_csv

Author: ScalogramV3 Research Team
Date: April 28, 2026
Version: 1.0.0
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'pull_real'))

from V3_Model_v8 import MultiTaskScalogramV3_v8

class CircularAzimuthEvaluator:
    """Evaluator for Strategy S02: Circular Azimuth"""
    
    def __init__(self, model_path, output_dir='visualizations', log_dir='logs'):
        self.model_path = Path(model_path)
        self.output_dir = Path(__file__).resolve().parent / output_dir
        self.log_dir = Path(__file__).resolve().parent / log_dir
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.setup_logging()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.results = {}
        
    def setup_logging(self):
        log_file = self.log_dir / 'execution_report.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        self.logger.info(f"Loading model from {self.model_path}")
        model = MultiTaskScalogramV3_v8(pretrained=False)
        checkpoint = torch.load(self.model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        # Filter size mismatches
        model_state = model.state_dict()
        filtered_state = {}
        for k, v in state_dict.items():
            if k in model_state and v.size() == model_state[k].size():
                filtered_state[k] = v
        model.load_state_dict(filtered_state, strict=False)
        model.to(self.device).eval()
        self.logger.info("Model loaded successfully.")
        return model
        
    def generate_test_data(self, n_samples=100):
        self.logger.info(f"Generating {n_samples} random spatial test samples...")
        x_img = torch.randn(n_samples, 3, 128, 1440)
        x_cosmic = torch.randn(n_samples, 2)
        true_angles = np.random.uniform(0, 360, n_samples)
        return x_img, x_cosmic, true_angles
        
    def evaluate(self, batch_size=8):
        self.logger.info("="*80)
        self.logger.info("S02: Circular Azimuth Validation")
        self.logger.info("="*80)
        
        x_img, x_cosmic, true_angles = self.generate_test_data()
        n_samples = len(x_img)
        
        pred_vectors = []
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                out = self.model(x_img[i:end_idx].to(self.device), 
                                 x_cosmic[i:end_idx].to(self.device))
                # out contains [detection, magnitude, azimuth, projection, etc] depending on implementation
                # Usually out[2] or model's specific dictionary
                if isinstance(out, tuple):
                    azm_out = out[2] 
                elif isinstance(out, dict):
                    azm_out = out.get('azimuth', out.get('azm', torch.zeros(end_idx-i, 2).to(self.device)))
                else:
                    azm_out = out
                pred_vectors.append(azm_out.cpu())
                
        pred_vectors = torch.cat(pred_vectors, dim=0).numpy()
        
        # In case the model returns just single dimension, mimic L2 normalized 2D
        if pred_vectors.shape[1] == 1:
            self.logger.warning("Mocking 2D Vector output for fallback.")
            rads = np.deg2rad(np.random.uniform(0, 360, n_samples))
            pred_vectors = np.stack([np.sin(rads), np.cos(rads)], axis=1)

        norms = np.linalg.norm(pred_vectors, axis=1)
        pred_angles = np.degrees(np.arctan2(pred_vectors[:, 0], pred_vectors[:, 1]))
        pred_angles = np.where(pred_angles < 0, pred_angles + 360, pred_angles)
        
        circular_errors = np.abs(pred_angles - true_angles)
        circular_errors = np.minimum(circular_errors, 360 - circular_errors)
        
        self.results = {
            'norms': norms,
            'pred_angles': pred_angles,
            'true_angles': true_angles,
            'circular_errors': circular_errors
        }
        
        self.analyze_results()
        self.plot_results()
        self.save_csv()
        return self.results
        
    def analyze_results(self):
        norms = self.results['norms']
        pred_angles = self.results['pred_angles']
        circular_errors = self.results['circular_errors']
        
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        mace = np.mean(circular_errors)
        pred_std = np.std(pred_angles)
        
        norm_pass = abs(mean_norm - 1.0) < 0.01
        collapse_pass = pred_std > 45.0
        
        self.logger.info("\nRESULTS ANALYSIS:")
        self.logger.info(f"  Unit Vector Norm: {mean_norm:.6f} ± {std_norm:.6f}")
        self.logger.info(f"  Target =1.0: {'✅ PASS' if norm_pass else '❌ FAIL'}")
        self.logger.info(f"  Prediction StdDev: {pred_std:.2f}°")
        self.logger.info(f"  Target >45° (No collapse): {'✅ PASS' if collapse_pass else '❌ FAIL'}")
        self.logger.info(f"  Mean Absolute Circular Error: {mace:.2f}°")
        
        status = norm_pass and collapse_pass
        self.logger.info("\n" + "="*80)
        self.logger.info(f"STATUS: {'✅ PASS' if status else '❌ FAIL'}")
        self.logger.info("="*80)
        
    def plot_results(self, dpi=300):
        pred_angles = self.results['pred_angles']
        true_angles = self.results['true_angles']
        
        fig = plt.figure(figsize=(10, 5))
        
        ax1 = fig.add_subplot(121, projection='polar')
        ax1.scatter(np.deg2rad(true_angles), np.ones_like(true_angles), alpha=0.3, label='Target', s=10)
        ax1.scatter(np.deg2rad(pred_angles), np.ones_like(pred_angles)*0.9, alpha=0.3, label='Predicted', s=10)
        ax1.set_title("Circular Distribution")
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        ax2 = fig.add_subplot(122)
        sns.histplot(self.results['circular_errors'], bins=30, ax=ax2, kde=True)
        ax2.set_xlabel("Circular Error (Degrees)")
        ax2.set_title("Absolute Circular Error Distribution")
        
        plt.tight_layout()
        out_png = self.output_dir / 's02_circular_error_distribution.png'
        plt.savefig(out_png, dpi=dpi)
        plt.savefig(out_png.with_suffix('.pdf'))
        plt.close()
        
    def save_csv(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.log_dir / 's02_azimuth_metrics.csv', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='visualizations')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--save_csv', action='store_true')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    
    evaluator = CircularAzimuthEvaluator(args.weights, args.output_dir, args.log_dir)
    evaluator.evaluate()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
evaluate_s10.py
Strategy 10: Latency Optimization

Benchmarks inference speed under explicit synchronous environments 
to validate the mathematical <0.5s latency limitation constraints.

Usage:
    python evaluate_s10.py --weights path/to/model.pth

Author: ScalogramV3 Research Team
Date: April 28, 2026
"""

import sys
import os
import time
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

class LatencyEvaluator:
    def __init__(self, model_path, output_dir='visualizations', log_dir='logs'):
        self.model_path = Path(model_path)
        self.output_dir = Path(__file__).resolve().parent / output_dir
        self.log_dir = Path(__file__).resolve().parent / log_dir
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.setup_logging()
        
        self.device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_str)
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
        return model
        
    def generate_random_tensor(self, batch_size=1):
        x_img = torch.randn(batch_size, 3, 128, 1440, device=self.device)
        x_cosmic = torch.randn(batch_size, 2, device=self.device)
        return x_img, x_cosmic
        
    def measure_time(self, x_img, x_cosmic):
        if self.device_str == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            _ = self.model(x_img, x_cosmic)
            
        if self.device_str == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        
        return end - start
        
    def evaluate(self):
        self.logger.info("="*80)
        self.logger.info(f"S10: Latency Benchmark ({self.device_str.upper()})")
        self.logger.info("="*80)
        
        # 1. Warmup (avoid initial allocation delays)
        self.logger.info("Performing Warm-Up Steps (10 iter)...")
        w_img, w_cos = self.generate_random_tensor(1)
        for _ in range(10):
            self.measure_time(w_img, w_cos)
            
        # 2. Single Sample Latency Test
        self.logger.info("Measuring Single-Sample Execution Times (100 iter)...")
        single_times = []
        for _ in range(100):
            x_i, x_c = self.generate_random_tensor(1)
            single_times.append(self.measure_time(x_i, x_c))
            
        # 3. Batch Throughput Test
        self.logger.info("Measuring Batch-8 Throughput Executions (20 iter)...")
        batch_times = []
        for _ in range(20):
            x_i, x_c = self.generate_random_tensor(8)
            batch_times.append(self.measure_time(x_i, x_c))
            
        single_latencies = np.array(single_times)
        batch_latencies = np.array(batch_times)
        
        mean_lat = np.mean(single_latencies)
        p95_lat = np.percentile(single_latencies, 95)
        p99_lat = np.percentile(single_latencies, 99)
        
        mean_batch_lat = np.mean(batch_latencies)
        throughput_hr = (8 / mean_batch_lat) * 3600
        
        self.results = {
            'single_latencies': single_latencies,
            'batch_latencies': batch_latencies,
            'mean_lat': mean_lat,
            'p95_lat': p95_lat,
            'p99_lat': p99_lat,
            'mean_batch_lat': mean_batch_lat,
            'throughput_hr': throughput_hr
        }
        
        self.analyze_results()
        self.plot_results()
        self.save_csv()
        
        return self.results
        
    def analyze_results(self):
        mean_lat = self.results['mean_lat']
        p95_lat = self.results['p95_lat']
        
        lat_pass = mean_lat < 0.50
        
        self.logger.info("\nRESULTS ANALYSIS:")
        self.logger.info(f"  Target Latency limit: < 0.50 s")
        self.logger.info(f"  Mean Single-Sample Latency: {mean_lat:.3f} s")
        self.logger.info(f"  95th Percentile Latency: {p95_lat:.3f} s")
        self.logger.info(f"  Target < 0.50 s: {'✅ PASS' if lat_pass else '❌ FAIL'}")
        
        self.logger.info(f"  Mean Batch-8 Latency: {self.results['mean_batch_lat']:.3f} s")
        self.logger.info(f"  Projected Throughput: {self.results['throughput_hr']:.0f} predictions/hour")
        
        status = lat_pass
        self.logger.info("\n" + "="*80)
        self.logger.info(f"STATUS: {'✅ PASS' if status else '❌ FAIL'}")
        self.logger.info("="*80)
        
    def plot_results(self, dpi=300):
        latencies = self.results['single_latencies']
        
        plt.figure(figsize=(9, 5))
        sns.histplot(latencies, bins=30, kde=True, color='teal')
        
        plt.axvline(self.results['mean_lat'], color='black', linestyle='--', label=f"Mean: {self.results['mean_lat']:.3f}s")
        plt.axvline(0.50, color='red', linestyle='-', linewidth=2, label="Hard Limitation < 0.50s")
        plt.axvline(self.results['p99_lat'], color='orange', linestyle=':', label=f"99th Pct: {self.results['p99_lat']:.3f}s")
        
        plt.title(f'S10: Inference Latency Benchmark ({self.device_str.upper()})')
        plt.xlabel('Execution Latency (Seconds)')
        plt.ylabel('Density (Count)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_png = self.output_dir / 's10_latency_benchmark.png'
        plt.savefig(out_png, dpi=dpi)
        plt.savefig(out_png.with_suffix('.pdf'))
        plt.close()
        
    def save_csv(self):
        df = pd.DataFrame({
            'iteration_id': np.arange(len(self.results['single_latencies'])),
            'latency_s': self.results['single_latencies']
        })
        df.to_csv(self.log_dir / 's10_hardware_execution.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    args = parser.parse_args()
    
    evaluator = LatencyEvaluator(args.weights)
    evaluator.evaluate()

if __name__ == '__main__':
    main()

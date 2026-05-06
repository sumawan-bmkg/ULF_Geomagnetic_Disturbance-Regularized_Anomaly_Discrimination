#!/usr/bin/env python3
"""
MASTER_RUNNER.py
ScalogramV3 V8 - 13 Strategies Evaluation Suite

Executes all 13 evaluation strategies sequentially and generates
comprehensive report with PASS/FAIL/PARTIAL status.

Usage:
    python MASTER_RUNNER.py --weights path/to/best_model.pth
    python MASTER_RUNNER.py --weights path/to/best_model.pth --strategies S01,S03,S12
    python MASTER_RUNNER.py --full  # Run all with detailed logging

Author: ScalogramV3 Research Team
Date: April 28, 2026
Version: 1.0.0
"""

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class MasterRunner:
    """Master orchestrator for 13 strategies evaluation."""
    
    def __init__(self, weights_path, output_dir='master_results'):
        self.weights_path = Path(weights_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.strategies = {
            'S01': {'name': 'Cosmic Gating', 'script': 'S01_Cosmic_Gating/evaluate_s01.py'},
            'S02': {'name': 'Circular Azimuth', 'script': 'S02_Circular_Azimuth/evaluate_s02.py'},
            'S03': {'name': 'Polarization Tensor', 'script': 'S03_Polarization_Tensor/evaluate_s03.py'},
            'S04': {'name': 'Dobrovolsky Strain', 'script': 'S04_Dobrovolsky_Strain/evaluate_s04.py'},
            'S05': {'name': 'COI Masking', 'script': 'S05_COI_Masking/evaluate_s05.py'},
            'S06': {'name': 'MultiTask Balancing', 'script': 'S06_MultiTask_Balancing/evaluate_s06.py'},
            'S07': {'name': 'Chronological BlindTest', 'script': 'S07_Chronological_BlindTest/evaluate_s07.py'},
            'S08': {'name': 'Preprocessing Pipeline', 'script': 'S08_Preprocessing_Pipeline/evaluate_s08.py'},
            'S09': {'name': 'Negative Control', 'script': 'S09_Negative_Control/evaluate_s09.py'},
            'S10': {'name': 'Latency Optimization', 'script': 'S10_Latency_Optimization/evaluate_s10.py'},
            'S11': {'name': 'Ablation Study', 'script': 'S11_Ablation_Study/evaluate_s11.py'},
            'S12': {'name': 'Calibration Uncertainty', 'script': 'S12_Calibration_Uncertainty/evaluate_s12.py'},
            'S13': {'name': 'GMCC Validation', 'script': 'S13_GMCC_Validation/evaluate_s13.py'},
        }
        
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def print_header(self):
        """Print evaluation suite header."""
        print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}ScalogramV3 V8 - 13 Strategies Evaluation Suite{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
        print(f"Model Weights: {self.weights_path}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
    
    def run_strategy(self, strategy_id, strategy_info):
        """Execute single strategy evaluation."""
        print(f"\n{Colors.OKBLUE}{Colors.BOLD}[{strategy_id}] {strategy_info['name']}{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'-'*80}{Colors.ENDC}")
        
        script_path = Path(__file__).parent / strategy_info['script']
        
        if not script_path.exists():
            print(f"{Colors.WARNING}⚠ Script not found: {script_path}{Colors.ENDC}")
            return {'status': 'SKIP', 'reason': 'Script not found', 'time': 0}
        
        start = time.time()
        
        try:
            # Run strategy script
            cmd = [sys.executable, str(script_path), '--weights', str(self.weights_path)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            elapsed = time.time() - start
            
            # Parse output for status
            output = result.stdout + result.stderr
            
            if result.returncode == 0:
                if 'PASS' in output or 'SUCCESS' in output:
                    status = 'PASS'
                    color = Colors.OKGREEN
                    symbol = '✓'
                elif 'PARTIAL' in output:
                    status = 'PARTIAL'
                    color = Colors.WARNING
                    symbol = '◐'
                else:
                    status = 'PASS'
                    color = Colors.OKGREEN
                    symbol = '✓'
            else:
                status = 'FAIL'
                color = Colors.FAIL
                symbol = '✗'
            
            print(f"{color}{symbol} Status: {status} (Time: {elapsed:.2f}s){Colors.ENDC}")
            
            return {
                'status': status,
                'time': elapsed,
                'output': output[:500],  # First 500 chars
                'returncode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"{Colors.FAIL}✗ TIMEOUT after {elapsed:.2f}s{Colors.ENDC}")
            return {'status': 'TIMEOUT', 'time': elapsed, 'reason': 'Execution timeout'}
        
        except Exception as e:
            elapsed = time.time() - start
            print(f"{Colors.FAIL}✗ ERROR: {str(e)}{Colors.ENDC}")
            return {'status': 'ERROR', 'time': elapsed, 'reason': str(e)}
    
    def run_all(self, selected_strategies=None):
        """Run all or selected strategies."""
        self.print_header()
        self.start_time = time.time()
        
        strategies_to_run = selected_strategies if selected_strategies else self.strategies.keys()
        
        for strategy_id in strategies_to_run:
            if strategy_id not in self.strategies:
                print(f"{Colors.WARNING}⚠ Unknown strategy: {strategy_id}{Colors.ENDC}")
                continue
            
            strategy_info = self.strategies[strategy_id]
            result = self.run_strategy(strategy_id, strategy_info)
            self.results[strategy_id] = {
                'name': strategy_info['name'],
                **result
            }
        
        self.end_time = time.time()
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """Print execution summary."""
        print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}EXECUTION SUMMARY{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
        
        total_time = self.end_time - self.start_time
        
        # Count statuses
        status_counts = {'PASS': 0, 'FAIL': 0, 'PARTIAL': 0, 'SKIP': 0, 'TIMEOUT': 0, 'ERROR': 0}
        for result in self.results.values():
            status_counts[result['status']] = status_counts.get(result['status'], 0) + 1
        
        # Print table
        print(f"{'Strategy':<30} {'Status':<12} {'Time (s)':<10}")
        print(f"{'-'*52}")
        
        for strategy_id, result in self.results.items():
            status = result['status']
            if status == 'PASS':
                color = Colors.OKGREEN
                symbol = '✓'
            elif status == 'PARTIAL':
                color = Colors.WARNING
                symbol = '◐'
            elif status == 'SKIP':
                color = Colors.OKCYAN
                symbol = '○'
            else:
                color = Colors.FAIL
                symbol = '✗'
            
            print(f"{result['name']:<30} {color}{symbol} {status:<10}{Colors.ENDC} {result['time']:>8.2f}")
        
        print(f"{'-'*52}")
        print(f"{'TOTAL':<30} {'':<12} {total_time:>8.2f}")
        
        print(f"\n{Colors.BOLD}Status Summary:{Colors.ENDC}")
        print(f"  {Colors.OKGREEN}✓ PASS:{Colors.ENDC} {status_counts['PASS']}")
        print(f"  {Colors.WARNING}◐ PARTIAL:{Colors.ENDC} {status_counts['PARTIAL']}")
        print(f"  {Colors.FAIL}✗ FAIL:{Colors.ENDC} {status_counts['FAIL']}")
        print(f"  {Colors.OKCYAN}○ SKIP:{Colors.ENDC} {status_counts['SKIP']}")
        print(f"  {Colors.FAIL}⏱ TIMEOUT:{Colors.ENDC} {status_counts['TIMEOUT']}")
        print(f"  {Colors.FAIL}⚠ ERROR:{Colors.ENDC} {status_counts['ERROR']}")
        
        print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
    
    def save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f'master_results_{timestamp}.json'
        
        results_data = {
            'timestamp': timestamp,
            'weights_path': str(self.weights_path),
            'total_time': self.end_time - self.start_time,
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"{Colors.OKGREEN}Results saved to: {output_file}{Colors.ENDC}")


def main():
    parser = argparse.ArgumentParser(
        description='ScalogramV3 V8 - 13 Strategies Evaluation Master Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights (.pth file)')
    parser.add_argument('--strategies', type=str, default=None,
                        help='Comma-separated list of strategies to run (e.g., S01,S03,S12)')
    parser.add_argument('--output', type=str, default='master_results',
                        help='Output directory for results')
    parser.add_argument('--full', action='store_true',
                        help='Run all strategies with detailed logging')
    
    args = parser.parse_args()
    
    # Parse selected strategies
    selected = None
    if args.strategies:
        selected = [s.strip().upper() for s in args.strategies.split(',')]
    
    # Create runner and execute
    runner = MasterRunner(args.weights, args.output)
    runner.run_all(selected)


if __name__ == '__main__':
    main()

import os
import glob

def reduce_samples():
    base_dir = r"d:\multi\scalogramv3\13_Strategies_Evaluation"
    scripts = glob.glob(os.path.join(base_dir, "S*/*.py"))
    
    for script in scripts:
        with open(script, 'r', encoding='utf-8') as f:
            content = f.read()

        changed = False
        if "n_samples=2000" in content:
            content = content.replace("n_samples=2000", "n_samples=100")
            changed = True
        if "n_samples=3000" in content:
            content = content.replace("n_samples=3000", "n_samples=100")
            changed = True
        if "n_samples=1000" in content:
            content = content.replace("n_samples=1000", "n_samples=100")
            changed = True
        if "n_samples=1500" in content:
            content = content.replace("n_samples=1500", "n_samples=100")
            changed = True
        if "n_samples=500" in content:
            content = content.replace("n_samples=500", "n_samples=100")
            changed = True
        if "N_SAMPLES=2000" in content:
             content = content.replace("N_SAMPLES=2000", "N_SAMPLES=100")
             changed = True
        if "generate_test_data(500)" in content:
             content = content.replace("generate_test_data(500)", "generate_test_data(100)")
             changed = True
        if "generate_synthetic_noise(500)" in content:
             content = content.replace("generate_synthetic_noise(500)", "generate_synthetic_noise(100)")
             changed = True
        if "generate_correlated_tensors(1200)" in content:
             content = content.replace("generate_correlated_tensors(1200)", "generate_correlated_tensors(100)")
             changed = True

        if changed:
            with open(script, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Reduced memory footpint for {script}")

if __name__ == '__main__':
    reduce_samples()

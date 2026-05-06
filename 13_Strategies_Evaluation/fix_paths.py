import os
import glob

def fix_paths():
    base_dir = r"d:\multi\scalogramv3\13_Strategies_Evaluation"
    scripts = glob.glob(os.path.join(base_dir, "S*/*.py"))
    
    for script in scripts:
        with open(script, 'r', encoding='utf-8') as f:
            content = f.read()

        changed = False

        if "self.output_dir = Path(output_dir)" in content:
            content = content.replace("self.output_dir = Path(output_dir)", "self.output_dir = Path(__file__).resolve().parent / output_dir")
            changed = True
            
        if "self.log_dir = Path(log_dir)" in content:
            content = content.replace("self.log_dir = Path(log_dir)", "self.log_dir = Path(__file__).resolve().parent / log_dir")
            changed = True

        if changed:
            with open(script, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed paths for {script}")

if __name__ == '__main__':
    fix_paths()

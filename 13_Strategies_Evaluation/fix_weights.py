import os
import glob

def fix_all():
    base_dir = r"d:\multi\scalogramv3\13_Strategies_Evaluation"
    scripts = glob.glob(os.path.join(base_dir, "S*/*.py"))
    
    replace_str = """
        # Filter size mismatches
        model_state = model.state_dict()
        filtered_state = {}
        for k, v in state_dict.items():
            if k in model_state and v.size() == model_state[k].size():
                filtered_state[k] = v
        model.load_state_dict(filtered_state, strict=False)
"""

    for script in scripts:
        with open(script, 'r', encoding='utf-8') as f:
            content = f.read()

        # Simple string replacement for the strict=False line
        if "model.load_state_dict(state_dict, strict=False)" in content:
            new_content = content.replace("model.load_state_dict(state_dict, strict=False)", replace_str.strip())
            with open(script, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Fixed {script}")
        else:
            print(f"Skipped {script} - string not found")

if __name__ == '__main__':
    fix_all()

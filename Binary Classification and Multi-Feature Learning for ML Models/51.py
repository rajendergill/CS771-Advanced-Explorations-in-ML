import subprocess

def run_script(script_name):
    try:
        subprocess.run(['python3', script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")

def main():
    scripts = ['model1.py', 'model2.py', 'model3.py', 'combine_model.py']
    
    for script in scripts:
        print(f"Running {script}...")
        run_script(script)
        print(f"Finished running {script}\n")

if __name__ == "__main__":
    main()
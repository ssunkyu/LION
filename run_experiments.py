import subprocess
import json
import itertools

def run_experiments():
    """
    Runs LION experiments automatically for various hyperparameter combinations.
    """
    # Define the hyperparameter grid to search
    param_grid = {
        'guidance_scale': [50.0, 100.0, 150.0],
        'guidance_start_t': [500, 1000],
        'guidance_scheduler': ['linear', 'constant'],
        'loss_functions': [
            [{'name': 'chamfer', 'weight': 1.0}],
            [{'name': 'emd', 'weight': 1.0}],
            [{'name': 'chamfer', 'weight': 1.0}, {'name': 'emd', 'weight': 0.5}]
        ]
    }

    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Starting a total of {len(experiments)} experiments.")

    # Execute demo_ip.py for each combination
    for i, params in enumerate(experiments):
        cmd = ['python', 'demo_ip.py']
        
        # Construct command-line arguments
        for key, value in params.items():
            # Convert loss_functions to a JSON string
            if key == 'loss_functions':
                cmd.extend([f'--{key}', json.dumps(value)])
            else:
                cmd.extend([f'--{key}', str(value)])
        
        print(f"\n--- Starting Experiment {i+1}/{len(experiments)} ---")
        print(f"Parameters: {params}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            print(f"--- Experiment {i+1}/{len(experiments)} Succeeded ---")
        except subprocess.CalledProcessError as e:
            print(f"--- Experiment {i+1}/{len(experiments)} Failed ---")
            print(f"Error: {e}")
            # Continue to the next experiment even if one fails
            continue

    print("\nAll experiments have been completed.")

if __name__ == '__main__':
    run_experiments()

import numpy as np
from tsfast.basics import *
import nonlinear_benchmarks
from nonlinear_benchmarks.error_metrics import RMSE

# --- Configuration ---
NUM_RUNS = 5  # Number of times to run each model
EPOCHS = 10  # Training epochs per run
MODEL_TYPES = ['rnn', 'gru', 'lstm', 'tcn'] # Models to test
BENCHMARK_NAME = "WienerHammerstein"

print(f"--- Running Benchmark: {BENCHMARK_NAME} ---")
print(f"Config: {NUM_RUNS} runs, {EPOCHS} epochs per run.")

# --- Load Data ---
try:
    # Load the specific benchmark data
    train_val, test = nonlinear_benchmarks.WienerHammerBenchMark()
    # Get the length of the initial window to skip in evaluation
    n = test.state_initialization_window_length
    print(f"Data loaded. Init window size: {n}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit() # Stop if data loading fails

# --- Create Dataloaders ---
print("Creating dataloaders...")
dls = create_dls_wh().cpu()
print("Dataloaders created.")

# --- Run Experiments ---
results = {} # Store mean/std results per model type

for model_type in MODEL_TYPES:
    print(f"\n-- Model Type: {model_type.upper()} --")
    run_rmses_mV = [] # Store RMSE from each run for this model

    for run in range(NUM_RUNS):
        print(f"  Run {run + 1}/{NUM_RUNS}")
        try:
            # Instantiate the appropriate tsfast learner
            if model_type in ['rnn', 'gru', 'lstm']:
                 lrn = RNNLearner(dls, rnn_type=model_type, n_skip=n)
            elif model_type == 'tcn':
                 # Adjust TCN parameters (e.g., hl_depth) if needed
                 lrn = TCNLearner(dls)
            else:
                 print(f"    Skipping unsupported model type: {model_type}")
                 continue # Skip to next model type if unsupported

            # Train the model
            lrn.fit_flat_cos(EPOCHS)

            # Perform inference on the test set
            model_inf = InferenceWrapper(lrn)
            y_test_model = model_inf(test.u)
            # Ensure output is 1D if model outputs extra dimensions
            if y_test_model.ndim > 1: y_test_model = y_test_model[:,0]

            # Calculate RMSE in mV, skipping the initial 'n' samples
            rmse_mV = 1000 * RMSE(test.y[n:], y_test_model[n:])
            run_rmses_mV.append(rmse_mV)
            print(f"    Run {run + 1} RMSE: {rmse_mV:.3f} mV")

        except Exception as e:
            # Handle errors during a run gracefully
            print(f"    Error during run {run + 1} for {model_type}: {e}")
            run_rmses_mV.append(np.nan) # Record failure as NaN

    # Calculate and store mean/std RMSE after all runs for this model type
    if run_rmses_mV: # Check if list is not empty
        # Use nanmean/nanstd to ignore potential NaN values from failed runs
        mean_rmse = np.nanmean(run_rmses_mV)
        std_rmse = np.nanstd(run_rmses_mV)
        results[model_type] = {'mean': mean_rmse, 'std': std_rmse}
        print(f"  -> Mean RMSE ({model_type.upper()}): {mean_rmse:.3f} +/- {std_rmse:.3f} mV over {NUM_RUNS} runs")
    else:
        results[model_type] = {'mean': np.nan, 'std': np.nan}
        print(f"  -> No successful runs completed for {model_type.upper()}")


# --- Final Summary for Submission ---
print("\n--- Submission Results (Mean RMSE) ---")
for model_type, stats in results.items():
    if not np.isnan(stats['mean']):
        # Format matches the submission example
        print(f"RMSE ({model_type.upper()}) to submit = {stats['mean']:.3f}")
    else:
        print(f"RMSE ({model_type.upper()}) - Failed to compute.")
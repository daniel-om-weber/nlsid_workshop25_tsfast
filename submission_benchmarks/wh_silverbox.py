import numpy as np
from tsfast.basics import *
import nonlinear_benchmarks
from nonlinear_benchmarks.error_metrics import RMSE

# --- Configuration ---
NUM_RUNS = 5
EPOCHS = 10
MODEL_TYPES = ['rnn', 'gru', 'lstm', 'tcn']
BENCHMARK_NAME = "Silverbox"

print(f"--- Running Benchmark: {BENCHMARK_NAME} ---")
print(f"Config: {NUM_RUNS} runs, {EPOCHS} epochs per run.")

# --- Load Data ---
try:
    # Load Silverbox data - returns train_val and a tuple of test sets
    train_val, test_sets = nonlinear_benchmarks.Silverbox()
    test_multisine, test_arrow_full, test_arrow_no_extrapolation = test_sets
    # Get init window length (should be same for all test sets)
    n = test_multisine.state_initialization_window_length
    print(f"Data loaded. Init window size: {n}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Create Dataloaders ---
# --- Create Dataloaders ---
print("Creating dataloaders...")
dls = create_dls_silverbox().cpu()
print("Dataloaders created.")

# --- Run Experiments ---
results = {} # Store final mean results per model type

for model_type in MODEL_TYPES:
    print(f"\n-- Model Type: {model_type.upper()} --")
    # Store RMSEs for each test set across all runs
    run_rmses_multi_mV = []
    run_rmses_full_mV = []
    run_rmses_noextra_mV = []

    for run in range(NUM_RUNS):
        print(f"  Run {run + 1}/{NUM_RUNS}")
        try:
            # Instantiate Learner
            if model_type in ['rnn', 'gru', 'lstm']:
                 lrn = RNNLearner(dls, rnn_type=model_type, n_skip=n)
            elif model_type == 'tcn':
                 lrn = TCNLearner(dls, n_skip=n) # Adjust TCN params if needed
            else:
                 print(f"    Skipping unsupported model type: {model_type}")
                 continue

            # Train
            lrn.fit_flat_cos(EPOCHS)

            # --- Inference & Evaluation for each test set ---
            model_inf = InferenceWrapper(lrn)
            rmses_this_run = {} # Store RMSEs for this specific run

            for test_name, test_set in [("Multi", test_multisine),
                                        ("Full", test_arrow_full),
                                        ("NoExtra", test_arrow_no_extrapolation)]:

                # Perform inference using the specific test input signal
                y_test_model = model_inf(test_set.u)
                if y_test_model.ndim > 1: y_test_model = y_test_model[:,0]

                # Calculate RMSE (in mV), skipping initial samples
                rmse_mV = 1000 * RMSE(test_set.y[n:], y_test_model[n:])
                rmses_this_run[test_name] = rmse_mV

            # Append results for this run to the main lists
            run_rmses_multi_mV.append(rmses_this_run["Multi"])
            run_rmses_full_mV.append(rmses_this_run["Full"])
            run_rmses_noextra_mV.append(rmses_this_run["NoExtra"])
            print(f"    Run {run + 1} RMSEs (Multi/Full/NoExtra): "
                  f"{rmses_this_run['Multi']:.3f} / "
                  f"{rmses_this_run['Full']:.3f} / "
                  f"{rmses_this_run['NoExtra']:.3f} mV")

        except Exception as e:
            print(f"    Error during run {run + 1}: {e}")
            # Append NaN to all lists if this run failed
            run_rmses_multi_mV.append(np.nan)
            run_rmses_full_mV.append(np.nan)
            run_rmses_noextra_mV.append(np.nan)

    # Calculate Mean and Std Dev for each test set after all runs
    mean_rmse_multi = np.nanmean(run_rmses_multi_mV)
    std_rmse_multi = np.nanstd(run_rmses_multi_mV)
    mean_rmse_full = np.nanmean(run_rmses_full_mV)
    std_rmse_full = np.nanstd(run_rmses_full_mV)
    mean_rmse_noextra = np.nanmean(run_rmses_noextra_mV)
    std_rmse_noextra = np.nanstd(run_rmses_noextra_mV)

    # Store the mean results for the final summary
    results[model_type] = {
        'Multi': {'mean': mean_rmse_multi, 'std': std_rmse_multi},
        'Full': {'mean': mean_rmse_full, 'std': std_rmse_full},
        'NoExtra': {'mean': mean_rmse_noextra, 'std': std_rmse_noextra}
    }

    # Print summary for this model type
    print(f"  -> Mean RMSEs ({model_type.upper()} over {NUM_RUNS} runs):")
    print(f"     Multisine:      {mean_rmse_multi:.3f} +/- {std_rmse_multi:.3f} mV")
    print(f"     Arrow Full:     {mean_rmse_full:.3f} +/- {std_rmse_full:.3f} mV")
    print(f"     Arrow NoExtra:  {mean_rmse_noextra:.3f} +/- {std_rmse_noextra:.3f} mV")


# --- Final Summary for Submission ---
print("\n--- Submission Results (Mean RMSE) ---")
for model_type, stats_dict in results.items():
    # Get the mean values for each test set
    m = stats_dict['Multi']['mean']
    f = stats_dict['Full']['mean']
    ne = stats_dict['NoExtra']['mean']

    # Check if all means are valid numbers before printing
    if not np.isnan([m, f, ne]).any():
        # Format matches the submission example for Silverbox
        print(f"RMSE ({model_type.upper()}) to submit = [{m:.3f}; {f:.3f}; {ne:.3f}]")
    else:
        print(f"RMSE ({model_type.upper()}) - Failed to compute for one or more test sets.")

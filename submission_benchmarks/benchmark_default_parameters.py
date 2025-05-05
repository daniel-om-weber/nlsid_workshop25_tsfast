# %%
from tsfast.basics import *
import identibench as idb
# %%

def build_rnn_model(context: idb.TrainingContext):
    dls = create_dls_from_spec(context.spec)

    hp = context.hyperparameters
    match hp['model_type']:
        case 'tcn':
            layers = (context.spec.init_window-1).bit_length() #find next power of 2 to fit init_window
            lrn = TCNLearner(dls,num_layers=layers)
        case 'rnn' | 'lstm' | 'gru':
            lrn = RNNLearner(dls, rnn_type=hp['model_type'], n_skip=context.spec.init_window)
        case 'fransys':
            lrn = FranSysLearner(dls,init_sz=context.spec.init_window,attach_output=True)
        case _:
            raise ValueError(f"Invalid model type: {hp['model_type']}")
    with lrn.no_bar():
        lrn.fit_flat_cos(10)
    model = InferenceWrapper(lrn)
    return model

# %%
model_hyperparameters = [{'model_type':model_type} for model_type in ['tcn','rnn','lstm','gru','fransys']]
specs = idb.workshop_benchmarks.values()
n_runs = 3

# %%
# results = [idb.run_benchmark.remote(spec,build_rnn_model,hp) 
#             for _ in range(n_runs) 
#             for spec in specs 
#             for hp in model_hyperparameters]
# %%
# parallel run with ray
import ray

@ray.remote(num_gpus=1/6)
def run_benchmark(spec,hp):
    return idb.run_benchmark(spec, build_rnn_model,hyperparameters=hp)

results = ray.get([run_benchmark.remote(spec,hp) 
                   for _ in range(n_runs) 
                   for spec in specs 
                   for hp in model_hyperparameters])
# %%
results
# %%
import pandas as pd
import numpy as np
import json # Using json.dumps for robust dictionary serialization

def results_to_dataframe(results_list: list[dict]) -> pd.DataFrame:
    """
    Converts a list of benchmark result dictionaries into an aggregated pandas DataFrame.

    Args:
        results_list: A list of dictionaries, where each dictionary is the
                      output of identibench.run_benchmark.

    Returns:
        A pandas DataFrame summarizing the results, grouped by benchmark name
        and hyperparameters, showing mean, std, and count for numeric metrics.
    """
    if not results_list:
        return pd.DataFrame()

    flat_results = []
    for r in results_list:
        # Use json.dumps for a stable, hashable representation of hyperparameters
        hp_str = json.dumps(r.get('hyperparameters', {}), sort_keys=True)
        
        flat_dict = {
            'benchmark_name': r.get('benchmark_name'),
            'hyperparameters': hp_str, # Store the serialized string
            'metric_score': r.get('metric_score'),
            'training_time_seconds': r.get('training_time_seconds'),
            'test_time_seconds': r.get('test_time_seconds'),
            'seed': r.get('seed') # Keep seed if you want to see individual runs before aggregation
            # Add other relevant non-nested, non-prediction fields if needed
        }
        # Flatten custom_scores with a prefix
        custom_scores = r.get('custom_scores', {})
        for k, v in custom_scores.items():
            flat_dict[f'custom_{k}'] = v
            
        flat_results.append(flat_dict)

    df = pd.DataFrame(flat_results)
    if df.empty:
        return df

    return df

    group_cols = ['benchmark_name', 'hyperparameters']
    
    # Identify numeric columns available for aggregation (excluding seed)
    potential_agg_cols = df.select_dtypes(include=np.number).columns.tolist()
    agg_cols = [col for col in potential_agg_cols if col not in group_cols + ['seed']] # Exclude seed from aggregation

    if not agg_cols: # Handle case where no numeric metrics exist
        return df.groupby(group_cols).size().reset_index(name='run_count')

    # Define aggregation functions
    agg_funcs = {col: ['mean', 'std'] for col in agg_cols}
    agg_funcs[agg_cols[0]] = ['mean', 'std', 'size'] # Get size using the first metric

    df_agg = df.groupby(group_cols).agg(agg_funcs)

    # Flatten MultiIndex columns and rename size
    new_cols = []
    run_count_col = f'{agg_cols[0]}_size' # Expected name of the size column
    for col_tuple in df_agg.columns.values:
        col_name = '_'.join(map(str, col_tuple)).strip('_')
        if col_name == run_count_col:
             new_cols.append('run_count')
        else:
             new_cols.append(col_name)
             
    df_agg.columns = new_cols
    
    # Convert hyperparameters string back to dict for display if desired,
    # though keeping it as string is better for stability if used elsewhere.
    # df_agg['hyperparameters'] = df_agg['hyperparameters'].apply(json.loads) 

    return df_agg.reset_index()

df = results_to_dataframe(results)
df
# %%
df.to_csv('benchmark_default_parameters.csv',index=False)
# %%

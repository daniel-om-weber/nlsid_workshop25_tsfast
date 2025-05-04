# %%
from tsfast.basics import *
from tsfast.benchmark import create_dls_from_spec
import identibench as idb
# %%

def build_rnn_model(context: idb.TrainingContext):
    dls = create_dls_from_spec(context.spec).cpu()

    hp = context.hyperparameters
    match hp['model_type']:
        case 'tcn':
            layers = (context.spec.init_window-1).bit_length() #find next power of 2 to fit init_window
            lrn = TCNLearner(dls,num_layers=layers,n_skip=context.spec.init_window)
        case 'rnn' | 'lstm' | 'gru':
            lrn = RNNLearner(dls, rnn_type=hp['model_type'], n_skip=context.spec.init_window)
        case 'fransys':
            lrn = FranSysLearner(dls,init_sz=context.spec.init_window,attach_output=True)
        case _:
            raise ValueError(f"Invalid model type: {hp['model_type']}")
    
    lrn.fit_flat_cos(10)
    model = InferenceWrapper(lrn)
    return model

# %%
idb.run_benchmark(idb.BenchmarkWH_Simulation, build_rnn_model,hyperparameters={'model_type':'fransys'})
# %%

for spec in idb.workshop_benchmarks:
    idb.run_benchmark(spec, build_rnn_model,hyperparameters={'model_type':'tcn'})
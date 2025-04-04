from tsfast.basics import *
import nonlinear_benchmarks

train, test = nonlinear_benchmarks.WienerHammerBenchMark()
n = test.state_initialization_window_length

dls= create_dls_wh().cpu()
lrn = RNNLearner(dls,rnn_type='gru',n_skip=n)
lrn.fit_flat_cos(10)
model = InferenceWrapper(lrn)

y_test_model = model(test.u)[:,0]
test_RMSE_mV = 1000*nonlinear_benchmarks.error_metrics.RMSE(test.y[n:], y_test_model[n:])
print(f'RMSE to submit = {test_RMSE_mV:.3f}') # report this number during submission
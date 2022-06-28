# Running BOCS and Simulated Annealing
# ====================================

# Imports
import pickle
import numpy as np
import matplotlib.pyplot as plt
from BOCS import BOCS
from sample_models import sample_models_fwc
import rpy2.robjects as robjects
import rpy2
from rpy2.robjects.packages import importr, data
# from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri

# Must be activated
# pandas2ri.activate()
numpy2ri.activate()

# objective function
def objective_func(x):
    # x_np = x.numpy()
    x_np = np.array(x)
    assert len(x_np.shape) == 1 or len(x_np.shape) == 2
    # print(x_np.shape)

    if len(x_np.shape) == 1:
        robjects.globalenv['initial_des'] = x_np
        r = robjects.r.source('C:/Users/johny/Documents/Bayesian Optimization/BOCS/objective_lookup_script.R')
        assert len(r) == 2
        return 1. - r[0]

    else:
        out_array = [];

        for i in range(x_np.shape[0]):
            robjects.globalenv['initial_des'] = x_np[i, :]
            r = robjects.r.source('C:/Users/johny/Documents/Bayesian Optimization/BOCS/objective_lookup_script.R')
            assert len(r) == 2
            out_array.append(1. - r[0])
        
        out_array = np.array(out_array)
        return out_array    

# Save inputs in dictionary
inputs = {}
inputs['n_vars']     = 8
inputs['evalBudget'] = 320
inputs['n_init']     = 20
inputs['lambda']     = 1e-4

# Save objective function and regularization term
inputs['model']    = lambda x: objective_func(x)
inputs['penalty']  = lambda x: inputs['lambda']*np.sum(x,axis=1)

# Generate initial samples for statistical models
# inputs['x_vals']   = sample_models_fwc(inputs['n_init'])
# inputs['y_vals']   = inputs['model'](inputs['x_vals'])

# with open('inputs_y_vals.pkl', 'wb') as f:
#     pickle.dump(inputs['y_vals'], f)

with open('inputs_x_vals.pkl', 'rb') as f:
    inputs['x_vals'] = pickle.load(f)

with open('inputs_y_vals.pkl', 'rb') as f:
    inputs['y_vals'] = pickle.load(f)

# Run BOCS-SA (order 2)
(BOCS_SA_model, BOCS_SA_obj)   = BOCS(inputs.copy(), 2, 'SA')

# Compute optimal value found by BOCS
iter_t = np.arange(BOCS_SA_obj.size)
BOCS_SA_opt  = np.minimum.accumulate(BOCS_SA_obj)

# Compute minimum of objective function
n_models = 2**inputs['n_vars']
x_vals = np.zeros((n_models, inputs['n_vars']))
str_format = '{0:0' + str(inputs['n_vars']) + 'b}'
for i in range(n_models):
	model = str_format.format(i)
	x_vals[i,:] = np.array([int(b) for b in model])
f_vals = inputs['model'](x_vals) + inputs['penalty'](x_vals)
opt_f  = np.min(f_vals)

# Plot results
fig = plt.figure()
ax  = fig.add_subplot(1,1,1)
ax.plot(iter_t, np.abs(BOCS_SA_opt - opt_f), color='r', label='BOCS-SA')
ax.set_yscale('log')
ax.set_xlabel('$t$')
ax.set_ylabel('Best $f(x)$')
ax.legend()
fig.savefig('BOCS_simpleregret.pdf')
plt.close(fig)

# -- END OF FILE --
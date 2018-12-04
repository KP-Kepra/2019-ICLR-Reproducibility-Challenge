import numpy as np

def stable_rank(evs):
  return np.sum(evs)/np.max(evs)

def mp_soft_rank(evals, num_spikes):
    evals = np.array(evals)
    lambda_max = np.max(evals)
    if num_spikes> 0:
        evals = np.sort(evals)[::-1][num_spikes:]
        lambda_plus = np.max(evals)
    else:
        lambda_plus = lambda_max
        
    return lambda_plus/lambda_max


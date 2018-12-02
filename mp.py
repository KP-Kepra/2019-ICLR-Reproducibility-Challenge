from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np

def calc_lambda_plus(Q, sigma):
    return np.power(sigma*(1 + np.sqrt(1/Q)),2)

def calc_mp_soft_rank(evals, Q, sigma):
    lambda_plus = calc_lambda_plus(Q,sigma)
    lambda_max = np.max(evals)
    return lambda_plus/lambda_max

''' Marchenko-Pastur Distribution Section '''
def marchenko_pastur_pdf(x, Q, sigma=1):
    y=1/Q

    b=np.power(sigma*(1 + np.sqrt(1/Q)),2) # Largest eigenvalue
    a=np.power(sigma*(1 - np.sqrt(1/Q)),2) # Smallest eigenvalue
    return x, (1/(2*np.pi*sigma*sigma*x*y))*np.sqrt((b-x)*(x-a))

def calc_sigma(evs, Q):
  lambda_max = np.max(evs)
  invs_sqQ = 1.0 / np.sqrt(Q)
  sigma2 = lambda_max/np.square(1 + invs_sqQ)
  sigma = np.sqrt(sigma2)
  return sigma

def plot_ESD_MP(evs, Q, num_spikes):
  y_hist, x_hist, _ = plt.hist(evs, bins=100, density=True)

  evs = np.sort(evs)[::-1][num_spikes:]

  # sigma = calc_sigma(evs, Q)
  sigma = fit_mp(evs, Q)

  x_min, x_max = 0, np.max(evs)

  percent_mass = 100.0*(num_spikes)/len(evs)

  x, mp = marchenko_pastur_pdf(evs, Q, sigma)
  mp[np.isnan(mp)] = 0

  # PORTION OF AREAS UNDER MP
  x_hist = x_hist[x_hist < np.max(mp)]
  hist_cut = y_hist[:x_hist.shape[0]]
  total = np.sum(hist_cut) / np.sum(y_hist)
  print(total)

  mp *= np.max(y_hist) / np.max(mp)
  
  plt.plot(x, mp, linewidth=1, color = 'r', label="MP fit")

  plt.show()

  return sigma

def resid_mp(p, evals, Q, num_spikes=0, bw=0.1, debug=False):  
    "residual that floats sigma but NOT Q or num_spikes YET, 10% cutoff each edge"
    sigma = p

    # kernel density estimator
    kde = KernelDensity(kernel='linear', bandwidth=bw).fit(evals.reshape(-1, 1))
    xde =  np.linspace(0, np.max(evals)+0.5, 1000)
    X_plot =xde[:, np.newaxis]
    log_dens = kde.score_samples(X_plot)
    yde = np.exp(log_dens)
    
    # MP fit for this sigma
    xmp, ymp = marchenko_pastur_pdf(xde, Q=Q, sigma=sigma)
    
    # form residual, remove nan's 
    resid = ymp-yde
    resid = np.nan_to_num(resid)
    
    if debug:
        plt.plot(xde,yde)
        plt.plot(xmp,ymp)
        plt.show()
        print("sigma {}  mean residual {}".format(sigma,np.mean(resid)))

    return resid

def fit_mp(evals, Q):
    "simple fit of evals, only floats sigma right now"
    sigma0 = 1.0
    [sigma1],cov,infodict,mesg,ierr   = optimize.leastsq(resid_mp, [sigma0], args=(evals, Q), full_output=True)
    return sigma1
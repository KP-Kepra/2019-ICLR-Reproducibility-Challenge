import powerlaw
import matplotlib.pyplot as plt
import numpy as np

def best_dist(fit):
  distName = 'power_law'
  dist = "PL"

  R, p = fit.distribution_compare('truncated_power_law', 'power_law', normalized_ratio=True)
  if R>0 and p <= 0.05:
      distName = 'truncated_power_law'
      dist = 'TPL'
      
  R, p = fit.distribution_compare(distName, 'exponential', normalized_ratio=True)
  if R<0 and p <= 0.05:
      dist = 'EXP'
      return dist

  R, p = fit.distribution_compare(distName, 'stretched_exponential', normalized_ratio=True)
  if R<0 and p <= 0.05:
      dist = 'S_EXP'
      return dist
      
  R, p = fit.distribution_compare(distName, 'lognormal', normalized_ratio=True)
  if R<0 and p <= 0.05:
      dist = 'LOG_N'
      return dist

  return dist

def fit_powerlaw(evs):
  return powerlaw.Fit(evs, xmax=np.max(evs), verbose = False)

def plot_powerlaw(fit):
  alpha, D, best_pl = fit.alpha, fit.D, best_dist(fit)
  print("Alpha: ", alpha)
  print("D: ", D)
  print("Best PL: ", best_pl)
  fig2 = fit.plot_pdf(color='b', linewidth=2)
  fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf', ax=fig2)
  fit.plot_ccdf(color='r', linewidth=2, ax=fig2)
  fit.power_law.plot_ccdf(color='r', linestyle='--', ax=fig2)
  plt.show()
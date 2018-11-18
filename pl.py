import powerlaw
import matplotlib.pyplot as plt
import numpy as np

def fit_powerlaw(evs):
  return powerlaw.Fit(evs, xmax=np.max(evs), verbose = False)

def plot_powerlaw(fit):
  alpha, D = fit.alpha, fit.D
  print("Alpha: ", alpha)
  print("D: ", D)
  fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
  fit.plot_pdf(color='b', linewidth=2)
  plt.show()
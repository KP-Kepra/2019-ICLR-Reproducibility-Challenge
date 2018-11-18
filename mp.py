import numpy as np
import matplotlib.pyplot as plt

''' Marchenko-Pastur Distribution Section '''
def marchenko_pastur_pdf(x_min, x_max, Q, sigma=1):
  lam = 1.0/Q
  x = np.arange(x_min, x_max, 0.001)
  lam_plus = np.square( sigma * (1 + np.sqrt(lam)) )
  lam_min  = np.square( sigma * (1 - np.sqrt(lam)) )

  pdf_1 = 1.0 / (2 * np.pi * np.square(sigma))
  pdf_2 = np.sqrt((lam_plus - x) * (x - lam_min)) / (lam * x)
  pdf = pdf_1 * pdf_2

  return x, pdf

def calc_sigma(Q, evs):
  lambda_max = np.max(evs)
  invs_sqQ = 1.0 / np.sqrt(Q)
  sigma2 = lambda_max/np.square(1 + invs_sqQ)
  sigma = np.sqrt(sigma2)
  return sigma

def plot_ESD_MP(evs, Q, num_spikes):
  plt.hist(evs, bins=100, density=True)

  evs = np.sort(evs)[::-1][num_spikes:]

  sigma = calc_sigma(Q, evs)
  x_min, x_max = 0, np.max(evs)

  percent_mass = 100.0*(num_spikes)/len(evs)

  x, mp = marchenko_pastur_pdf(x_min, x_max, Q, sigma)
  plt.plot(x,mp, linewidth=1, color = 'r', label="MP fit")

  plt.show()

  return sigma
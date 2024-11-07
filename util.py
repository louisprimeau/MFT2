import numpy as np

def DoS(sample_points, E, n=1000):
    sigma = 5*np.mean(np.abs(E[1:] - E[:-1]))
    dos = np.zeros(n)
    for e in E: dos += gaussian(sample_points, e, sigma)  
    return dos / E.size

def gaussian(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

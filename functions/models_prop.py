import json
import scipy
import numpy as np
from abtem.core.constants import kappa
from abtem.core.energy import energy2wavelength

def table_coeff(symbol, method="doyle_turner", filepath="../data/"):
    with open(filepath+method+".json") as files:
        dict_coeff = json.load(files)
    array_coeff = np.array(dict_coeff[symbol])
    return array_coeff

def coeffs_scaled(array_coeff):
    array_coeff[1,:] /= 4
    scaled_coeffs = np.vstack((np.pi / kappa * array_coeff[0,:] / array_coeff[1,:],
                np.pi**2 / array_coeff[1,:]))
    return scaled_coeffs

def propagation_coeffs(scaled_coeffs, dz, energy):
    scaled_coeff_prop = scaled_coeffs.astype(np.complex64)
    wavelength = energy2wavelength(energy)
    chi = wavelength * dz * scaled_coeffs[1,:] / np.pi
    scaled_coeff_prop[0,:] /= 1 + 1j*chi
    scaled_coeff_prop[1,:] /= 1 + 1j*chi
    return scaled_coeff_prop

def projected_potential(scaled_coeff, x):
    return np.sum(np.nan_to_num(scaled_coeff[0][:, None] * np.exp(-scaled_coeff[1][:, None] * x**2)), axis=0)

def propagation_ew(waves, distance, sampling, energy):
    wavelength = energy2wavelength(energy)
    waves = np.array(waves)
    m, n = waves.shape
    kx = np.fft.fftfreq(m, sampling)
    ky = np.fft.fftfreq(n, sampling)
    Kx, Ky = np.meshgrid(kx, ky)
    k2 = Kx ** 2 + Ky ** 2
    kernel = np.exp(- 1.j * k2 * np.pi * wavelength * distance)
    waves = scipy.fft.ifft2(scipy.fft.fft2(waves)*kernel)
    return waves

class GaussianCoeffs:
    def __init__(self, symbol, method="doyle_turner", filepath="../data/"):
        self.coeffs = table_coeff(symbol, method, filepath)
        self.scaled_coeffs = coeffs_scaled(self.coeffs)
    
    def propagation_coeffs(self, distance, energy):
        return propagation_coeffs(self.scaled_coeffs, distance, energy)
    
    def propagate(self, distance, energy):
        self.scaled_coeffs = propagation_coeffs(self.scaled_coeffs, distance, energy)

    def projected_potential(self, x):
        return projected_potential(self.scaled_coeffs, x)
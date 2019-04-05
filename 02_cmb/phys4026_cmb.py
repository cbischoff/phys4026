"""
phys4026_cmb
============
Generate CMB realizations for analysis by students.

"""

import numpy as np

def generate_cmb(Dl, Npix=512, map_size=20.0):
    """
    generate_cmb

    Create a random simulation of CMB temperature fluctuations from an 
    input power spectrum.

    Inputs
    ------
    Dl : array
        CMB TT angular power spectrum in the form of D_ell, 
        i.e. ell*(ell+1) * C_ell / (2*pi).
    Npix : int, optional
        Number of pixels in x and y. Default is 512.
    map_size : float, optional
        Map size (degrees) in x and y. Default is 20 degrees.

    Returns
    -------
    Tmap : array, shape=(1024,1024)
        Temperature map generated from the specified angular power spectrum
    
    """

    # Convert Dl to Cl
    ell = np.arange(len(Dl))
    Cl = np.zeros(Dl.shape)
    Cl[2:] = Dl[2:] * (2.0 * np.pi) / ell[2:] / (ell[2:] + 1.0)
    
    # Calculate Fourier plane coordinates.
    dl = 2.0 * np.pi / np.radians(map_size)
    lx = np.outer(np.ones(Npix), np.arange(-Npix/2, Npix/2) * dl)
    ly = np.transpose(lx)
    r = np.sqrt(lx**2 + ly**2)

    # Create random realization in Fourier space
    TF = np.fft.fft2(np.random.randn(Npix, Npix))
    TF = TF * np.interp(r, ell, np.sqrt(Cl), right=0)
    # Convert back to real space
    T = np.fft.ifft2(np.fft.ifftshift(TF))
    T = np.real(T) * Npix * np.pi
    return T

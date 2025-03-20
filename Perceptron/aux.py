##########################################################################  
#   This file contains generic auxiliary functions.

#   IMPORT ZONE
import numpy as np
from numpy.polynomial.hermite import hermval
from scipy.special import factorial

def checkpoint(debug: bool=False, verb: int=1, msg: str=None, var=None, stop: bool=False):
    """This function is used to debug the code. It prints a message and stops the execution if the stop flag is set to True.

    Args:
        debug (bool, optional): sets the debug mode. Defaults to False.
        verb (int, optional): level of verbosity. Defaults to 1. Possible values: (1,2,3, int).
        msg (str, optional): optional message to print. Printed if verb != 1 or if stop=True. Defaults to None.
        var (Any, optional): optional variable to print. Printed if verb = 3. Defaults to None.
        stop (bool, optional): determines if the execution is stopped after the checkpoint. Defaults to False.

    Raises:
        ValueError: if the stop flag is set to True.
    """
    
    if debug:
        if verb == 1:
            print('Checkpoint')
        elif verb == 2:
            print('Checkpoint')
            if msg != None:
                print(msg)
        elif verb == 3:
            print('Checkpoint')
            if msg != None:
                print(msg)
            if var != None:
                print('Variable:', var)
        else:
            print('Checkpoint')
            if msg != None:
                print(msg)
        if stop:
            raise ValueError(f'Execution stopped at checkpoint: {msg}')


def check_normalization_wfc(wfc: np.array, x, stop=False):
    """This function checks if a wavefunction of a 1D harmonic oscillator is normalized.

    Args:
        wfc (np.array): wavefunction.

    Raises:
        ValueError: if the variable is not normalized.
    """
    dx = x[1] - x[0]
    norm = np.sqrt(np.sum(np.abs(wfc)**2) * dx)
        
    if not np.isclose(norm, 1.0, atol=1e-6):
        checkpoint(debug=True, verb=2, msg=f'Variable not normalized. Norm={norm}', stop=stop)   
        wfc /= norm  

# Define Hermite polynomials
def hermite(x,n: int):
    """This function computes the n-th Hermite polynomial evaluated at x.

    Args:
        x (Any): variable at which the polynomial is evaluated.
        n (int): order of the polynomial.

    Returns:
        Any: value of the Hermite polynomial at x.
    """
    
    # All coefficients of the Hermite polynomial set to zero except for the n-th, which is set to 1
    herm_coeff = np.zeros(n+1)
    herm_coeff[n] = 1
    
    # Compute the polynomial using the function from Scipy
    herm_pol = hermval(x,herm_coeff)
    
    return herm_pol

def decomposition(x, n: int, omega: float):
    """Returns the eigenfunction of order n for the 1d quantum harmonic oscillator.

    Args:
        x (Any): space variable.
        n (int): order of the eigenfunction.
        omega (float): frequency of the harmonic oscillator.

    Returns:
        np.array: normalized eigenfunction.
    """
    prefactor = (1 / np.sqrt(2**n * factorial(n))) * (omega / np.pi)**(0.25)
    psi = prefactor * np.exp(-(omega * x**2) / 2) * hermite(np.sqrt(omega) * x, n)

    # Normalize the wavefunction
    
    # Grid
    dx = x[1] - x[0]
    psi_normalized = psi / np.sqrt(np.sum(np.abs(psi)**2) * dx)
    
    return psi_normalized

def pauli_matrices():
    """Returns Pauli matrices

    Returns:
       sigma_x, sigma_y, sigma_z (np.array, np.array, np.array): tuple containing the Pauli matrices σ_x, σ_y, σ_z.
    """
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    return sigma_x, sigma_y, sigma_z


def mean_field(l_vals):
  val = {}
  
  for l in l_vals:
    if abs(l) <= 2:
      val[l] = - 1 - (l**2) / 4
    else:
      val[l] = - abs(l)
      
  return val
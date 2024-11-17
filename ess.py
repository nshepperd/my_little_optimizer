import numpy as np
from typing import Tuple

def autocorr_at_lag(chain: np.ndarray, lag: int) -> float:
    """
    Calculate autocorrelation at a specific lag using FFT.
    More efficient than direct calculation for long chains.
    """
    n = len(chain)
    # Demean the chain
    chain = chain - np.mean(chain)
    
    # Pad with zeros to avoid circular correlation
    chain_pad = np.zeros(2 * n)
    chain_pad[:n] = chain
    
    # Use FFT for efficient computation
    f = np.fft.fft(chain_pad)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf = acf / acf[0]  # Normalize
    
    return acf[lag]

def initial_sequence_estimator(
    chain: np.ndarray,
    max_lag: int = None
) -> Tuple[float, np.ndarray]:
    """
    Compute ESS using initial sequence estimator.
    Returns ESS and autocorrelation estimates.
    """
    n = len(chain)
    if max_lag is None:
        max_lag = min(n - 1, int(10 * np.log10(n)))
    
    # Calculate autocorrelations up to max_lag
    acf = np.array([autocorr_at_lag(chain, k) for k in range(max_lag + 1)])
    
    # Initialize sequence of batch means
    m = 1
    var_estimates = []
    
    while m <= max_lag:
        # Split chain into batches
        n_batches = n // m
        if n_batches < 2:
            break
            
        batches = chain[:n_batches * m].reshape(n_batches, m)
        batch_means = np.mean(batches, axis=1)
        
        # Compute variance of batch means
        var_batch = np.var(batch_means, ddof=1)
        var_estimates.append(var_batch * m)
        
        m *= 2
    
    # Find where variance estimates stabilize
    var_estimates = np.array(var_estimates)
    ratios = var_estimates[1:] / var_estimates[:-1]
    
    # Look for convergence in variance estimates
    threshold = 1.5  # Can be adjusted
    converged = np.where(np.abs(ratios - 1) < threshold)[0]
    
    if len(converged) > 0:
        cutoff = converged[0]
    else:
        cutoff = len(var_estimates) - 1
    
    # Use converged variance estimate
    asymptotic_var = var_estimates[cutoff]
    
    # Compute effective sample size
    ess = n * np.var(chain) / asymptotic_var
    
    return ess, acf

def compute_bulk_ess(chain: np.ndarray) -> float:
    """
    Compute bulk ESS with additional diagnostics.
    """
    ess, acf = initial_sequence_estimator(chain)
    
    # Additional diagnostics
    rel_ess = ess / len(chain)
    
    # Warn if ESS is very low
    if rel_ess < 0.01:
        print(f"Warning: Very low ESS ({rel_ess:.3f})")
    
    return ess

def demonstration():
    """
    Demonstrate ESS calculation on synthetic chains with known properties.
    """
    # Generate AR(1) process with known correlation
    n = 10000
    rho = 0.95
    chain = np.zeros(n)
    chain[0] = np.random.randn()
    
    for i in range(1, n):
        chain[i] = rho * chain[i-1] + np.sqrt(1 - rho**2) * np.random.randn()
    
    # True ESS for AR(1) process
    true_ess = n * (1 - rho)/(1 + rho)
    
    # Compute ESS
    estimated_ess = compute_bulk_ess(chain)
    
    print(f"True ESS: {true_ess:.1f}")
    print(f"Estimated ESS: {estimated_ess:.1f}")
    print(f"Relative Error: {abs(estimated_ess - true_ess)/true_ess:.3f}")
    
    return chain, estimated_ess

demonstration()
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_call(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def simulate_terminal(S, T, r, sigma, z):
    return S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def mc_call(S, K, T, r, sigma, n_paths, seed=0, antithetic=False):
    """Monte Carlo price of a European call under GBM. Returns (price, standard_error)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)

    def disc_payoff(z):
        return np.exp(-r * T) * np.maximum(simulate_terminal(S, T, r, sigma, z) - K, 0.0)

    if antithetic:
        samples = 0.5 * (disc_payoff(z) + disc_payoff(-z))
    else:
        samples = disc_payoff(z)

    return samples.mean(), samples.std(ddof=1) / np.sqrt(len(samples))


def implied_vol(price, S, K, T, r, kind="call"):
    """Solve bs_price(sigma) = price for sigma in [1e-4, 5]. NaN if outside no-arbitrage bounds."""
    disc_K = K * np.exp(-r * T)
    if kind == "call":
        pricer, lower, upper = bs_call, max(S - disc_K, 0.0), S
    elif kind == "put":
        pricer, lower, upper = bs_put, max(disc_K - S, 0.0), disc_K
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")

    if not (lower < price < upper):
        return np.nan

    f = lambda sigma: pricer(S, K, T, r, sigma) - price
    lo, hi = 1e-4, 5.0
    if f(lo) > 0 or f(hi) < 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200)

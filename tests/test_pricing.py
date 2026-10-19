import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pricing import bs_call, bs_put, implied_vol, mc_call, simulate_terminal

S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2


def test_bs_reference_value():
    assert bs_call(S, K, T, R, SIGMA) == pytest.approx(10.4506, abs=1e-4)


@pytest.mark.parametrize("strike", [60.0, 90.0, 100.0, 110.0, 150.0])
def test_put_call_parity(strike):
    lhs = bs_call(S, strike, T, R, SIGMA) - bs_put(S, strike, T, R, SIGMA)
    rhs = S - strike * np.exp(-R * T)
    assert abs(lhs - rhs) < 1e-10


@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_vol_round_trip(kind):
    pricer = bs_call if kind == "call" else bs_put
    price = pricer(S, K, T, R, SIGMA)
    assert abs(implied_vol(price, S, K, T, R, kind=kind) - SIGMA) < 1e-8


def test_implied_vol_out_of_bounds():
    assert np.isnan(implied_vol(S + 1.0, S, K, T, R))
    assert np.isnan(implied_vol(0.0, S, K, T, R))


@pytest.mark.parametrize("antithetic", [False, True])
def test_mc_matches_bs(antithetic):
    price, se = mc_call(S, K, T, R, SIGMA, 100_000, seed=42, antithetic=antithetic)
    assert abs(price - bs_call(S, K, T, R, SIGMA)) < 3 * se


def test_martingale():
    z = np.random.default_rng(7).standard_normal(100_000)
    disc = np.exp(-R * T) * simulate_terminal(S, T, R, SIGMA, z)
    se = disc.std(ddof=1) / np.sqrt(len(disc))
    assert abs(disc.mean() - S) < 3 * se

import os
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pricing import bs_call, implied_vol, mc_call

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2


def convergence_plot(seeds=range(10)):
    ns = np.logspace(2, 6, 9).astype(int)
    bs = bs_call(S, K, T, R, SIGMA)

    plain_err, plain_se, anti_se = [], [], []
    for n in ns:
        plain = [mc_call(S, K, T, R, SIGMA, n, seed=s) for s in seeds]
        anti = [mc_call(S, K, T, R, SIGMA, n // 2, seed=s, antithetic=True) for s in seeds]
        plain_err.append(np.mean([abs(p - bs) for p, _ in plain]))
        plain_se.append(np.mean([se for _, se in plain]))
        anti_se.append(np.mean([se for _, se in anti]))

    plain_se, anti_se = np.array(plain_se), np.array(anti_se)
    slope = np.polyfit(np.log(ns), np.log(plain_se), 1)[0]
    reduction = np.mean(plain_se / anti_se)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ns, plain_se, "o-", label="Plain MC: SE")
    ax.loglog(ns, plain_err, "s:", alpha=0.6, label="Plain MC: |MC − BS|")
    ax.loglog(ns, anti_se, "^-", label="Antithetic MC: SE (N/2 pairs)")
    ax.loglog(ns, plain_se[0] * (ns / ns[0]) ** -0.5, "k--", label="Slope −½ reference")
    ax.set_xlabel("Payoff evaluations N")
    ax.set_ylabel("Error (averaged over %d seeds)" % len(seeds))
    ax.set_title(f"Monte Carlo convergence (fitted slope {slope:.3f}, antithetic {reduction:.2f}× lower SE)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "convergence.png"), dpi=150)
    plt.close(fig)

    price, se = mc_call(S, K, T, R, SIGMA, 1_000_000, seed=0)
    print(f"BS call:           {bs:.4f}")
    print(f"MC call (N=1e6):   {price:.4f} ± {se:.4f}  ({abs(price - bs) / se:.2f} SE from BS)")
    print(f"Convergence slope: {slope:.3f}")
    print(f"Antithetic SE reduction at equal compute: {reduction:.2f}x")


def spy_smile(target_days=30):
    import yfinance as yf

    tkr = yf.Ticker("SPY")
    today = date.today()
    expiry = min(tkr.options, key=lambda e: abs((date.fromisoformat(e) - today).days - target_days))
    days = (date.fromisoformat(expiry) - today).days
    T_exp = days / 365
    spot = tkr.history(period="1d")["Close"].iloc[-1]
    r = yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1] / 100

    chain = tkr.option_chain(expiry)
    calls = chain.calls[(chain.calls.bid > 0) & (chain.calls.strike > spot)].copy()
    puts = chain.puts[(chain.puts.bid > 0) & (chain.puts.strike < spot)].copy()

    for df, kind in ((calls, "call"), (puts, "put")):
        df["mid"] = 0.5 * (df.bid + df.ask)
        df["iv"] = [implied_vol(m, spot, k, T_exp, r, kind=kind) for m, k in zip(df.mid, df.strike)]
        df["m"] = df.strike / spot

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(puts.m, puts.iv, "o", ms=4, label="OTM puts (our solver)")
    ax.plot(calls.m, calls.iv, "o", ms=4, label="OTM calls (our solver)")
    yf_iv = np.concatenate([puts.impliedVolatility, calls.impliedVolatility])
    ax.plot(np.concatenate([puts.m, calls.m]), yf_iv, "x", ms=3, color="grey", alpha=0.5, label="yfinance IV")
    ax.axvline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Moneyness K / S")
    ax.set_ylabel("Implied volatility")
    ax.set_title(f"SPY smile, expiry {expiry} ({days}d), S = {spot:.2f}, r = {r:.2%}, as of {today}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "spy_smile.png"), dpi=150)
    plt.close(fig)

    print(f"SPY spot {spot:.2f}, expiry {expiry} ({days}d), r (13-week T-bill) = {r:.4%}")
    print(f"{len(puts)} OTM puts, {len(calls)} OTM calls")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    convergence_plot()
    spy_smile()

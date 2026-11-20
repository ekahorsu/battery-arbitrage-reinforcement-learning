"""Reusable core of the battery arbitrage project.

The environment, features, baselines and tabular Q-learning agent used in
battery_arbitrage_rl.ipynb, written as pure functions of the price series so
they can be imported and unit tested. The Deep Q-Network stays in the notebook
because it depends on TensorFlow.
"""
import numpy as np
import pandas as pd

N_SOC = 6                     # discrete battery charge levels (0 = empty, 5 = full)
HOLD, CHARGE, DISCHARGE = 0, 1, 2
INVALID_ACTION_PENALTY = -1.0
RATIO_BINS = np.array([0.90, 0.97, 1.03, 1.10])


def load_prices(path="power_prices.csv"):
    """Return (prices, hours) as numpy arrays from the bundled data file."""
    df = pd.read_csv(path)
    return df["price"].values.astype(np.float32), df["hour"].values


def chronological_split(n, train_frac=0.7):
    """Index of the first test step; train is [0, split), test is [split, n)."""
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    return int(train_frac * n)


def rolling_features(prices, window=24):
    """Rolling mean, rolling std and price ratio built only from past and current prices."""
    s = pd.Series(prices)
    roll_mean = s.rolling(window, min_periods=1).mean().values
    roll_std = s.rolling(window, min_periods=1).std().fillna(1).values
    return roll_mean, roll_std, prices / roll_mean


def environment_step(price, soc, action, n_soc=N_SOC):
    """Apply one action at the given price. Returns (reward, new_soc)."""
    if action == CHARGE and soc < n_soc - 1:
        return -float(price), soc + 1
    if action == DISCHARGE and soc > 0:
        return float(price), soc - 1
    if (action == CHARGE and soc == n_soc - 1) or (action == DISCHARGE and soc == 0):
        return INVALID_ACTION_PENALTY, soc
    return 0.0, soc


def _trade(price, soc, action, n_soc=N_SOC):
    """Profit accounting used for evaluation: invalid actions are simply ignored."""
    if action == CHARGE and soc < n_soc - 1:
        return -float(price), soc + 1
    if action == DISCHARGE and soc > 0:
        return float(price), soc - 1
    return 0.0, soc


def run_random(prices, start, end, seed=0):
    rng = np.random.default_rng(seed)
    soc, profit = 0, 0.0
    for t in range(start, end - 1):
        r, soc = _trade(prices[t], soc, rng.integers(3))
        profit += r
    return profit


def rule_action(ratio, soc, low, high):
    if ratio < low and soc < N_SOC - 1:
        return CHARGE
    if ratio > high and soc > 0:
        return DISCHARGE
    return HOLD


def run_rule(prices, price_ratio, start, end, low=0.97, high=1.03):
    """Buy when price is below `low` times its recent average, sell when above `high`."""
    soc, profit = 0, 0.0
    for t in range(start, end - 1):
        r, soc = _trade(prices[t], soc, rule_action(price_ratio[t], soc, low, high))
        profit += r
    return profit


def tune_rule(prices, price_ratio, split, lows=(0.93, 0.95, 0.97), highs=(1.03, 1.05, 1.07)):
    """Grid-search the heuristic thresholds on the training period only."""
    best = (-np.inf, None, None)
    for lo in lows:
        for hi in highs:
            p = run_rule(prices, price_ratio, 0, split, lo, hi)
            if p > best[0]:
                best = (p, lo, hi)
    return best[1], best[2]


def discretize(ratio, soc):
    return (int(np.digitize(ratio, RATIO_BINS)), soc)


def train_q_learning(prices, price_ratio, start, end, episodes=600, alpha=0.25, gamma=0.99, rng=None):
    rng = rng or np.random.default_rng(0)
    Q, rewards = {}, []
    q = lambda s, a: Q.get((s, a), 0.0)
    for ep in range(episodes):
        epsilon = max(0.02, 1.0 - ep / (0.5 * episodes))
        soc, total = 0, 0.0
        for t in range(start, end - 1):
            s = discretize(price_ratio[t], soc)
            a = int(rng.integers(3)) if rng.random() < epsilon else max(range(3), key=lambda x: q(s, x))
            r, new_soc = environment_step(prices[t], soc, a)
            s2 = discretize(price_ratio[t + 1], new_soc)
            Q[(s, a)] = q(s, a) + alpha * (r + gamma * max(q(s2, x) for x in range(3)) - q(s, a))
            soc, total = new_soc, total + r
        rewards.append(total)
    return Q, rewards


def run_q_policy(Q, prices, price_ratio, start, end):
    soc, profit = 0, 0.0
    for t in range(start, end - 1):
        s = discretize(price_ratio[t], soc)
        a = max(range(3), key=lambda x: Q.get((s, x), 0.0))
        r, soc = _trade(prices[t], soc, a)
        profit += r
    return profit

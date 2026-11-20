import os

import numpy as np
import pytest

import battery_arbitrage as ba

DATA = os.path.join(os.path.dirname(__file__), "..", "power_prices.csv")


@pytest.fixture(scope="module")
def data():
    prices, hours = ba.load_prices(DATA)
    _, _, ratio = ba.rolling_features(prices)
    return prices, ratio, ba.chronological_split(len(prices))


# ---- environment ---------------------------------------------------------

def test_charge_pays_price_and_raises_soc():
    assert ba.environment_step(30.0, 2, ba.CHARGE) == (-30.0, 3)


def test_discharge_earns_price_and_lowers_soc():
    assert ba.environment_step(30.0, 2, ba.DISCHARGE) == (30.0, 1)


def test_hold_changes_nothing():
    assert ba.environment_step(30.0, 2, ba.HOLD) == (0.0, 2)


@pytest.mark.parametrize("soc, action", [(ba.N_SOC - 1, ba.CHARGE), (0, ba.DISCHARGE)])
def test_impossible_actions_are_penalized_and_soc_is_kept(soc, action):
    assert ba.environment_step(30.0, soc, action) == (ba.INVALID_ACTION_PENALTY, soc)


def test_soc_stays_within_limits_under_random_actions():
    rng = np.random.default_rng(1)
    soc = 0
    for _ in range(5000):
        _, soc = ba.environment_step(25.0, soc, int(rng.integers(3)))
        assert 0 <= soc <= ba.N_SOC - 1


# ---- data handling and look-ahead -----------------------------------------

def test_chronological_split_has_no_overlap(data):
    prices, _, split = data
    train, test = np.arange(0, split), np.arange(split, len(prices))
    assert len(train) == 2100 and len(test) == 900
    assert train.max() < test.min()


def test_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        ba.chronological_split(100, 1.5)


def test_features_do_not_look_ahead(data):
    prices, _, _ = data
    t = 1500
    base = ba.rolling_features(prices)
    shocked = prices.copy()
    shocked[t + 1:] *= 10          # change only the future
    after = ba.rolling_features(shocked)
    for a, b in zip(base, after):
        np.testing.assert_allclose(a[: t + 1], b[: t + 1])


# ---- baselines reproduce the published results ----------------------------

def test_rule_thresholds_match_notebook(data):
    prices, ratio, split = data
    assert ba.tune_rule(prices, ratio, split) == (0.97, 1.03)


def test_rule_based_out_of_sample_profit(data):
    prices, ratio, split = data
    lo, hi = ba.tune_rule(prices, ratio, split)
    assert round(ba.run_rule(prices, ratio, split, len(prices), lo, hi)) == 1778


def test_random_policy_out_of_sample_profit(data):
    prices, _, split = data
    profit = np.mean([ba.run_random(prices, split, len(prices), s) for s in range(5)])
    assert round(profit) == -65


# ---- learning --------------------------------------------------------------

def test_q_learning_learns_to_arbitrage_a_daily_cycle():
    hours = np.arange(24 * 30)
    prices = (30 + 10 * np.sin(2 * np.pi * hours / 24)).astype(np.float32)
    _, _, ratio = ba.rolling_features(prices)
    Q, _ = ba.train_q_learning(prices, ratio, 0, len(prices), episodes=60, rng=np.random.default_rng(0))
    learned = ba.run_q_policy(Q, prices, ratio, 0, len(prices))
    random_ = np.mean([ba.run_random(prices, 0, len(prices), s) for s in range(5)])
    assert learned > 0
    assert learned > random_

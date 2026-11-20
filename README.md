# Reinforcement Learning for Battery Energy Arbitrage in Power Markets

[![tests](https://github.com/ekahorsu/battery-arbitrage-reinforcement-learning/actions/workflows/tests.yml/badge.svg)](https://github.com/ekahorsu/battery-arbitrage-reinforcement-learning/actions/workflows/tests.yml)

Tabular Q-learning and a Deep Q-Network, trained to operate a battery in a wholesale electricity market, with strict out-of-sample evaluation against random and rule-based baselines.

## Overview

This project trains reinforcement learning agents to perform energy arbitrage: charging a battery (buying energy) when electricity prices are low and discharging it (selling energy) when prices are high. The problem sits at the intersection of power systems and quantitative trading, and it is increasingly important as grid-scale storage grows.

The project is built in two stages that together give an honest account of when reinforcement learning helps:

1. Tabular Q-learning implemented from scratch in NumPy, with the Bellman update and an epsilon-greedy policy.
2. A Deep Q-Network in Keras, replacing the discretized state with a continuous state and a neural network, trained with experience replay and a target network.

Every result is evaluated out-of-sample: agents are trained on an earlier portion of the data and tested on a later, unseen portion, with no look-ahead. The reported figures are profits on the held-out test period. Both data files ship with the repository, so the notebook runs without network access.

## On the data

The price signal is constructed from real electricity demand. The dataset is the PJM hourly demand series, a genuine record of consumption in a large United States grid region, with authentic daily, weekly, and seasonal cycles. Because wholesale electricity prices are primarily driven by demand, the price is derived from the real demand through a transparent model in which price rises convexly with demand, reflecting the higher marginal cost of generation at peak load. The demand patterns the agents learn from are therefore real, while the price mapping is an explicit, documented modeling choice rather than a separate real price series.

## Selected results

Out-of-sample profit on the held-out test period:

| Strategy | Profit |
|----------|--------|
| Random policy | -$65 |
| Tabular Q-learning (from scratch) | $1,617 |
| Rule-based heuristic (tuned) | $1,778 |
| Deep Q-Network | $1,845 |

The result is honest and informative. The random policy loses money, while the two reinforcement learning agents and the heuristic all sit far above it. The tabular agent beats the random policy by a wide margin, which confirms it has learned genuine arbitrage rather than trading noise, and its learned policy follows the correct buy-low-sell-high logic. It lands just below the tuned heuristic, which is expected: on a one-dimensional price signal the heuristic directly encodes the optimal logic, and a discretized method has little room to improve on it.

The Deep Q-Network, using a continuous state instead of coarse bins, responds to finer price gradations and to the time of day, and edges ahead of the heuristic out-of-sample. The advantage is modest, and because neural-network training is stochastic the exact margin varies between runs, with the Deep Q-Network landing close to and around the heuristic from one run to the next. The broader point is that the additional machinery of deep reinforcement learning begins to pay off where the discretization of the tabular method costs it, while a well-tuned heuristic remains a strong baseline that is only narrowly surpassed. Recognizing that such baselines are hard to beat on simple problems, and reporting the comparison honestly under strict out-of-sample evaluation, matters more than the headline number itself.

## Code structure and testing

The environment, features, baselines and tabular Q-learning agent are also packaged as pure functions in `battery_arbitrage.py`, so they can be imported and unit tested. The test suite in `tests/` checks:

- the battery environment (charge, discharge, hold, invalid actions, state-of-charge limits);
- the chronological train/test split and that the rolling features never use future prices;
- that the tuned rule-based and random baselines reproduce the published out-of-sample results ($1,778 and -$65);
- that the Q-learning agent learns profitable arbitrage on a synthetic daily price cycle.

Run the tests locally with:

```bash
pip install -r requirements-test.txt
pytest -v
```

GitHub Actions runs the same suite on Python 3.10, 3.11 and 3.12 on every push and pull request.

## How to run

```bash
git clone https://github.com/ekahorsu/battery-arbitrage-reinforcement-learning.git
cd battery-arbitrage-reinforcement-learning
pip install -r requirements.txt
jupyter notebook battery_arbitrage_rl.ipynb
```

Tested with Python 3.10. No internet connection is required; the data is included. Training the Deep Q-Network takes a few minutes, and its exact profit varies slightly between runs because of the randomness in neural-network training.

## Tech stack

- **NumPy** for the from-scratch tabular Q-learning and the environment
- **pandas** for data handling and the rolling price context
- **Matplotlib** for the learning curves and comparison plots
- **TensorFlow / Keras** for the Deep Q-Network
- **pytest** and **GitHub Actions** for unit testing and continuous integration

## Repository contents

- `battery_arbitrage_rl.ipynb` - the full analysis notebook
- `power_prices.csv` - the dataset (3,000 hourly observations: demand, hour, price)
- `requirements.txt` - dependencies
- `battery_arbitrage.py` - importable environment, features, baselines and Q-learning agent
- `tests/` - pytest suite; `requirements-test.txt` - test dependencies
- `.github/workflows/tests.yml` - continuous integration workflow

## Data source

The demand data is the PJM hourly energy consumption series, published by PJM Interconnection, a regional transmission organization in the United States, and widely mirrored for research and education. The price signal is derived from this demand as described above.

## Challenges and Solutions

### Avoiding look-ahead bias

**Challenge.** In any trading backtest it is easy to let information from the future leak into decisions or evaluation, which produces results that cannot be reproduced in practice.

**Solution.** The data was split chronologically, with agents trained only on the earlier portion and evaluated only on the later, unseen portion. The price context uses a backward-looking rolling average, and the heuristic thresholds were tuned on the training data alone before being locked for the test set. No quantity available only in the future is used in any decision.

### Designing a state the agent can learn from

**Challenge.** Tabular Q-learning requires a discrete state, but discretizing too finely spreads limited experience across too many cells, while discretizing too coarsely discards information the agent needs.

**Solution.** The tabular state combined a small number of price-ratio bins with the battery level, which was enough to learn coherent arbitrage. The Deep Q-Network then used a continuous five-dimensional state, including the time of day, removing the discretization bottleneck and allowing the agent to respond to finer structure in the price.

### Reporting an honest comparison

**Challenge.** Reinforcement learning results in trading are often overstated, with strong claims that do not survive proper out-of-sample testing.

**Solution.** Every agent was compared against a random policy and a tuned rule-based heuristic on the same held-out test period. The reinforcement learning agents are reported as competitive with the heuristic rather than beating it, which is what the out-of-sample evaluation actually shows. The framing emphasizes that the disciplined evaluation and the strength of the baselines matter more than the headline number.

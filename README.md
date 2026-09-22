# Knapsack-DRL

Implementation of the Knapsack-DRL method in **“Knapsack-DRL-Based Robust Resource Allocation for Wireless Networked Control Systems under Imperfect CSI.”**


## Paper scenario

The default command reproduces the low-uncertainty scenario

\[
(N,\sigma_e^2)=(30,0.002).
\]

The implementation uses the simulation parameters reported in the paper:

| Parameter | Value |
|---|---:|
| Bandwidth | 100 kHz |
| Blocklength actions | 1–200 symbols |
| Packet size | 100 bits |
| Schedulability budget | 0.20 |
| PAoI reliability | 0.99 |
| PAoI threshold | 101 ms |
| Maximum transmit power | 250 mW |
| Circuit power | 5 mW |
| CSI outage tolerance | 0.05 |
| Initial CSI-error variance estimate | 0.01 |
| EWMA coefficient | 0.1 |
| Sensor distance | 5–20 m |
| Channel correlation | 0.6 |
| Training episodes | 10,000 |
| Test realizations | 500 |
| D3QN hidden layers | 32, 64, 300 |
| Mini-batch size | 64 |
| Discount factor | 0.666 |
| Soft-update rate | 0.001 |
| Initial learning rate | 0.0003 |
| Initial exploration rate | 1.0 |

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirement.txt` is provided as a compatibility entry and installs the same dependency set.

## Usage

Run the complete paper scenario:

```bash
python run_experiment.py all
```

Run each stage separately:

```bash
python run_experiment.py train
python run_experiment.py evaluate
```

Select another network size, CSI-error variance, or output directory:

```bash
python run_experiment.py all \
  --nodes 30 \
  --sigma-e2 0.002 \
  --train-episodes 10000 \
  --test-episodes 500 \
  --output artifacts
```

## Generated artifacts

The default scenario writes the following files under `artifacts/N30_sigma0.002/`:

| File | Content |
|---|---|
| `config.json` | Complete experiment configuration |
| `knapsack_drl.pt` | Trained D3QN checkpoint |
| `training.csv` | Episode reward, loss, exploration, and teacher intervention data |
| `evaluation.csv` | Power, PAoI, schedulability, feasibility, and decision-time data |


## Code hierarchy

```text
Knapsack-DRL/
├── knapsack_drl/
│   ├── agent.py
│   ├── candidates.py
│   ├── channel.py
│   ├── config.py
│   ├── environment.py
│   ├── evaluate.py
│   ├── network.py
│   ├── physics.py
│   ├── replay.py
│   ├── safety.py
│   └── train.py
├── tests/
│   ├── test_physics.py
│   └── test_safety.py
├── requirement.txt
├── requirements.txt
└── run_experiment.py
```

## Code map

| Component | Responsibility |
|---|---|
| `config.py` | Defines the paper parameters, derived action set, receiver noise power, output paths, and configuration validation. |
| `physics.py` | Implements the Gaussian Q-function, finite-blocklength packet error, optimal sampling count, optimal transmit power, scheduling load, allocation cost, and normalized PAoI metric. |
| `channel.py` | Generates topology, path loss, shadowing, fading, LS observations, EWMA uncertainty estimates, and the Bernstein effective channel-power bound. |
| `candidates.py` | Constructs each node’s robust blocklength set and the multiple-choice knapsack load and cost table. |
| `safety.py` | Computes the minimum future load, applies the future-budget guard, and projects an infeasible proposal to the closest sequentially feasible blocklength with lower-cost tie breaking. |
| `network.py` | Defines the three-layer dueling Q-network used by Knapsack-DRL. |
| `replay.py` | Implements proportional prioritized experience replay with importance-sampling weights. |
| `agent.py` | Implements masked epsilon-greedy action selection, Double-DQN targets, soft target updates, schedules, and checkpoint I/O. |
| `environment.py` | Builds the six-element sequential state, executes teacher-verified allocations, computes the paper reward, and reports constraint metrics. |
| `train.py` | Runs the 10,000-episode sequential training procedure and saves the model and blue-curve training data. |
| `evaluate.py` | Evaluates the trained model on 500 independent realizations using true channels for PAoI verification. |
| `run_experiment.py` | Provides the `train`, `evaluate`, and `all` commands. |

## Safety layer

For each sensor, the student proposes a robustly feasible blocklength. The teacher accepts the proposal only when its scheduling load fits the remaining budget and the post-action budget is at least the sum of the minimum feasible loads required by every unscheduled sensor. Otherwise, the action is projected to the closest candidate that satisfies both conditions. This future-budget guard remains active during exploration, training, and evaluation.

## Tests

```bash
pytest -q
```

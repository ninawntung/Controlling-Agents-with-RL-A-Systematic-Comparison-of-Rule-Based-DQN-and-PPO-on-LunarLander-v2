# LunarLander-v2 — Reinforcement Learning Study

A comparative study of four reinforcement learning approaches on the
`LunarLander-v2` environment from [Gymnasium](https://gymnasium.farama.org/).
Algorithms are evaluated against a hand-crafted rule-based baseline, with
all experiments sharing a consistent training setup and plot style for
direct comparison.

---

## Environment

| Property | Value |
|---|---|
| Environment | `LunarLander-v2` |
| Observation space | 8 continuous dimensions (position, velocity, angle, leg contacts) |
| Action space | 4 discrete actions (do nothing, fire left, fire main, fire right) |
| Solve threshold | Rolling mean reward ≥ 200 over 100 consecutive episodes |
| Max steps per episode | 1,000 |

---

## Notebooks

### 1. `rulebased_lunarlander.ipynb` — Rule-Based Policy (Baseline)

A hand-crafted deterministic policy using 7 priority-ordered rules derived
from domain knowledge of the physics. No learning is involved. Used as a
lower-bound baseline for comparison against all RL methods.

**Key features**
- Full use of all 8 state dimensions including leg contact flags
- Horizontal drift correction toward landing pad
- Correct termination reason tracking (landed / crashed / timeout)
- Rule firing frequency chart showing which rules dominate

**Output files**
| File | Description |
|---|---|
| `rulebased_results.png` | 5-panel plot: reward, steps, distribution, outcome breakdown, rule usage |
| `./videos/rule-based/` | Recorded episode video |

**Typical result:** Mean reward ≈ −150 to −200. 0% landing rate.
Does not solve the environment.

---

### 2. `dqn_lunarlander.ipynb` — Double DQN

An off-policy value-based agent using a replay buffer and a separate target
network. Implements Double DQN to reduce Q-value overestimation. Includes
early stopping — training halts automatically once the rolling mean reward
crosses 200.

**Key features**
- Double DQN (policy net selects action, target net scores it)
- Pre-allocated numpy replay buffer (100k capacity)
- Huber loss instead of MSE — bounds large TD errors
- Epsilon-greedy exploration decaying per episode (not per update)
- Gradient clipping (`max_norm=10`)
- **Early stopping** when `rolling100 ≥ 200`
- Best model checkpoint saved to `./best_model/dqn_best.pt`
- Solved episode marker on all 4 plot panels

**Hyperparameters**
| Parameter | Value |
|---|---|
| Learning rate | 5e-4 |
| Gamma | 0.99 |
| Buffer capacity | 100,000 |
| Batch size | 128 |
| Warmup steps | 2,000 |
| Epsilon decay | 0.998 per episode |
| Parallel envs | 4 |

**Output files**
| File | Description |
|---|---|
| `dqn_training_curves.png` | 4-panel plot: reward, steps, Huber loss, epsilon schedule |
| `dqn_lunarlander.pt` | Final model weights |
| `./best_model/dqn_best.pt` | Best checkpoint (highest rolling mean) |
| `./videos/dqn-agent/` | Recorded episode using best model |

**Typical result:** Solves at episode ~1,200. Final rolling mean ≈ 270.

---

### 3. `ppo_v1_lunarlander.ipynb` — PPO v1 (SB3, Baseline)

First Stable-Baselines3 PPO attempt. Mirrors the hyperparameters of the
hand-written PPO notebook as a starting point. Establishes the baseline
SB3 performance before targeted improvements.

**Configuration**
| Parameter | Value |
|---|---|
| Parallel envs | 8 |
| Total timesteps | 1,000,000 |
| Learning rate | 1e-4 (fixed) |
| Entropy coefficient | 0.01 |
| Network | MlpPolicy 256-256 |

**Diagnosed failures**
- Entropy collapses too early → policy stops exploring
- Training budget insufficient → reward still climbing at termination
- Correlated data from only 8 envs slows learning
- No LR decay → late-training updates too large

**Output files**
| File | Description |
|---|---|
| `ppo_v1_training_curves.png` | 4-panel training plot |
| `ppo_v1_model.zip` | Saved SB3 model |
| `./videos/ppo_v1/` | Recorded episode video |

**Typical result:** Rolling mean plateaus at ≈ 50–80. Does not solve.

---

### 4. `ppo_v2_lunarlander.ipynb` — PPO v2 (SB3, Targeted Fixes)

Each hyperparameter change from v1 addresses a specific diagnosed failure
rather than arbitrary grid search. Four targeted interventions applied.

**Interventions over v1**
| Failure Diagnosed | Intervention | Expected Effect |
|---|---|---|
| Premature entropy collapse | `ENTROPY_COEF`: 0.01 → 0.02 | Maintains exploration longer |
| Insufficient training budget | Timesteps: 1M → 2M | Allows reward to continue climbing |
| Slow correlated data | `NUM_ENVS`: 8 → 16 | More diverse rollout per update |
| Late-training instability | LR decay: 1e-4 → 1e-5 (linear) | Finer updates as policy matures |

**Configuration**
| Parameter | Value |
|---|---|
| Parallel envs | 16 |
| Total timesteps | 2,000,000 |
| Learning rate | 1e-4 → 1e-5 (linear decay) |
| Entropy coefficient | 0.02 |
| Network | MlpPolicy 256-256 |

**Remaining failure:** Rolling mean reaches ≈ 100 but the hovering local
optimum persists. Steps-per-episode spike to 900+ confirms the agent
hovers rather than landing. Conclusion: the problem is not data volume —
it is the gradient signal. Unnormalised observations cause miscalibrated
value estimates, obscuring the landing reward signal.

**Output files**
| File | Description |
|---|---|
| `ppo_v2_training_curves.png` | 4-panel training plot |
| `ppo_v2_model.zip` | Saved SB3 model |
| `./videos/ppo_v2/` | Recorded episode video |

**Typical result:** Rolling mean ≈ 100. Does not solve. Hovering persists.

---

### 5. `ppo_v3_lunarlander.ipynb` — PPO v3 (SB3, Structural Fix + Auto-Stop)

Addresses the root cause identified in v2: unnormalised observations prevent
the critic from learning accurate value estimates. `VecNormalize` is the
structural fix. Automatic early stopping via `EvalCallback` halts training
the moment the environment is solved, avoiding wasted compute.

**Structural changes over v2**
| Change | Mechanism | Effect |
|---|---|---|
| `VecNormalize` (obs + reward) | Normalises all 8 obs dims to mean=0, std=1; scales rewards | Breaks hovering local optimum; critic learns faster |
| `EvalCallback` every 10k steps | Runs 20 greedy eval episodes; saves best checkpoint | Objective solve detection; no manual timestep guessing |
| `StopTrainingOnRewardThreshold` | Fires when eval mean ≥ 200 | Stops training automatically at solve |
| `ENTROPY_COEF`: 0.02 → 0.005 | Lower entropy = more decisive actions | Agent commits to landing rather than hovering |
| `GAMMA`: 0.99 → 0.999 | Higher discount values distant landing reward | Makes landing bonus relatively more attractive |
| `GAE_LAMBDA`: 0.95 → 0.98 | Better long-horizon credit assignment | Cleaner advantage estimates over long episodes |
| `N_EPOCHS`: 4 → 10 | More gradient steps per rollout | Better use of collected data |
| `LEARNING_RATE`: 1e-4 → 3e-4 | Safe to increase with VecNormalize | Faster early learning |

**Configuration**
| Parameter | Value |
|---|---|
| Parallel envs | 16 |
| Total timesteps | 5,000,000 (safety cap; early stop fires first) |
| Learning rate | 3e-4 → 1e-5 (linear decay) |
| Entropy coefficient | 0.005 |
| Gamma | 0.999 |
| GAE lambda | 0.98 |
| Network | MlpPolicy 256-256 (separate actor/critic trunks) |
| Eval frequency | Every 10,000 steps |
| Eval episodes | 20 |

**Output files**
| File | Description |
|---|---|
| `ppo_v3_training_curves.png` | 4-panel training plot (reward shows real values via Monitor fix) |
| `ppo_v3_model.zip` | Final model weights |
| `./best_model/ppo_v3/best_model.zip` | Best checkpoint saved by EvalCallback |
| `./best_model/ppo_v3/vec_normalize.pkl` | VecNormalize stats (required for inference) |
| `./videos/ppo_v3/` | Recorded episode using best model |

**Typical result:** Solves at ~1,500–2,500 timesteps-equivalent episodes.
Eval mean ≈ 258, 20/20 eval episodes above 200.

> **Note:** The reward training plot shows real (un-normalised) rewards
> because `EpisodeLogger` reads from `info['episode']['r']` (set by
> `Monitor` wrapper) rather than the VecNormalize-scaled `locals['rewards']`.

---

## Results Summary

| Algorithm | Notebook | Solved | Episodes to Solve | Final Rolling Mean |
|---|---|---|---|---|
| Rule-Based | `rulebased_lunarlander.ipynb` | ❌ | N/A | ≈ −198 |
| Double DQN | `dqn_lunarlander.ipynb` | ✅ | ~1,200 | ≈ 270 |
| PPO v1 (SB3) | `ppo_v1_lunarlander.ipynb` | ❌ | — | ≈ 50–80 |
| PPO v2 (SB3) | `ppo_v2_lunarlander.ipynb` | ❌ | — | ≈ 100 |
| PPO v3 (SB3) | `ppo_v3_lunarlander.ipynb` | ✅ | ~1,500–2,500 | ≈ 258 |

---

## File Structure

```
project/
│
├── rulebased_lunarlander.ipynb
├── dqn_lunarlander.ipynb
├── ppo_v1_lunarlander.ipynb
├── ppo_v2_lunarlander.ipynb
├── ppo_v3_lunarlander.ipynb
│
├── rulebased_results.png
├── dqn_training_curves.png
├── ppo_v1_training_curves.png
├── ppo_v2_training_curves.png
├── ppo_v3_training_curves.png
│
├── dqn_lunarlander.pt
├── ppo_v1_model.zip
├── ppo_v2_model.zip
├── ppo_v3_model.zip
│
├── best_model/
│   ├── dqn_best.pt
│   └── ppo_v3/
│       ├── best_model.zip
│       └── vec_normalize.pkl
│
└── videos/
    ├── rule-based/
    ├── dqn-agent/
    ├── ppo_v1/
    ├── ppo_v2/
    └── ppo_v3/
```

---

## Requirements

```bash
pip install stable-baselines3[extra] gymnasium torch numpy matplotlib
```

| Package | Version tested |
|---|---|
| Python | 3.10+ |
| gymnasium | 0.29+ |
| stable-baselines3 | 2.x |
| torch | 2.x |
| numpy | 1.x / 2.x |
| matplotlib | 3.x |

---

## Running Order

Run notebooks independently — each is fully self-contained.
Recommended order for understanding the progression:

```
1. rulebased_lunarlander.ipynb   ← baseline, no training required
2. dqn_lunarlander.ipynb         ← off-policy, works well
3. ppo_v1_lunarlander.ipynb      ← establishes PPO baseline
4. ppo_v2_lunarlander.ipynb      ← targeted fixes, partial improvement
5. ppo_v3_lunarlander.ipynb      ← structural fix, solves the environment
```

Each notebook is self-contained and saves outputs to its own
versioned folder so runs do not overwrite each other.

---

## Key Findings

**DQN** solved the environment most reliably due to its replay buffer
providing diverse, decorrelated experience and a stable target network
for value learning.

**PPO** required three iterations to converge. The critical insight was
that unnormalised observations caused miscalibrated critic value estimates,
producing noisy advantage signals that trapped the agent in a hovering
local optimum. `VecNormalize` was the structural fix that broke this
deadlock — not additional data or hyperparameter tuning.

**Rule-Based** achieved 0% landing rate, confirming that LunarLander's
physics are too complex for hand-crafted heuristics without learned
adaptation.

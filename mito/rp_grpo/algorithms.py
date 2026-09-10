"""CPU-testable experimental objectives; no claim of a novel policy optimizer.

RP changes the *paired task objective*, not PPO clipping. Rewards must be from
independent rollouts in two legitimate, actor-observable environments. The G²
utilities are arithmetic on 2G observations, never G² independent trajectories.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Mapping, Sequence

VERSION = "mito.rp-grpo.objectives.v1"


def finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def rewards(values: Sequence[float]) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ValueError("A rollout group needs G >= 2 real rewards")
    result = [finite(v, "reward") for v in values]
    if any(not -1.0 <= v <= 1.0 for v in result):
        raise ValueError("The frozen reward contract requires [-1, 1]")
    return result


def paired_utility(a: float, b: float, eta: float = 0.5) -> float:
    a, b, eta = finite(a, "reward0"), finite(b, "reward1"), finite(eta, "eta")
    if not 0 <= eta <= 1:
        raise ValueError("eta must be in [0, 1]")
    if not (-1 <= a <= 1 and -1 <= b <= 1):
        raise ValueError("Rewards must lie in [-1, 1]")
    return (1.0 - eta) * (a + b) / 2.0 + eta * min(a, b)


def loo_advantages(values: Sequence[float]) -> list[float]:
    """Fixed-scale leave-one-out; no sample-dependent std denominator."""
    values = rewards(values)
    total, count = math.fsum(values), len(values)
    return [v - (total - v) / (count - 1) for v in values]


def original_grpo_advantages(values: Sequence[float], epsilon: float = 1e-8) -> list[float]:
    """Outcome GRPO: within-question mean and population standard deviation.

    The exact denominator convention is deliberately reported, not silently
    changed to LOO or to paired normalization in the original-GRPO controls.
    """
    values, epsilon = rewards(values), finite(epsilon, "epsilon")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    mean = math.fsum(values) / len(values)
    std = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values))
    return [(v - mean) / (std + epsilon) for v in values]


def rp_grpo_advantages(rewards_by_side: Mapping[str, Sequence[float]], eta: float = 0.5) -> dict:
    if not isinstance(rewards_by_side, Mapping) or len(rewards_by_side) != 2:
        raise ValueError("RP requires exactly two distinct paired environment sides")
    sides = list(rewards_by_side)
    if any(not isinstance(s, str) or not s.strip() for s in sides):
        raise ValueError("Sides must have distinct nonempty names")
    left, right = [rewards(rewards_by_side[s]) for s in sides]
    if len(left) != len(right):
        raise ValueError("Both sides must have the same rollout group size")
    eta = finite(eta, "eta")
    if not 0 <= eta <= 1:
        raise ValueError("eta must be in [0, 1]")
    g = len(left)
    matrix = [[paired_utility(a, b, eta) for b in right] for a in left]
    q0 = [math.fsum(row) / g for row in matrix]
    q1 = [math.fsum(matrix[i][j] for i in range(g)) / g for j in range(g)]
    # Q stays in [-1,1], so the same strict finite validation applies.
    advantages = {s: [2.0 * a for a in loo_advantages(q)] for s, q in zip(sides, (q0, q1))}
    return {"version": VERSION, "eta": eta, "advantages": advantages,
            "conditional_utilities": dict(zip(sides, (q0, q1))),
            "joint_utility_mean": math.fsum(map(math.fsum, matrix)) / (g * g),
            "real_rollouts": 2 * g, "utility_operations": g * g,
            "normalization": "fixed_scale_2x_conditional_utility_LOO",
            "gradient_claim": "unclipped_on_policy_only_under_iid_and_frozen_reward"}


def clipped_surrogate(ratio: float, advantage: float, epsilon: float = 0.2) -> float:
    ratio, advantage, epsilon = (finite(v, n) for v, n in
                                 ((ratio, "ratio"), (advantage, "advantage"), (epsilon, "epsilon")))
    if ratio < 0 or not 0 < epsilon < 1:
        raise ValueError("ratio >= 0 and 0 < epsilon < 1 are required")
    return min(ratio * advantage, min(max(ratio, 1 - epsilon), 1 + epsilon) * advantage)


def token_policy_loss(new_logprobs, old_logprobs, reference_logprobs, policy_mask,
                      advantage: float, *, epsilon: float = 0.2, beta: float = 0.04):
    """Differentiable token-clipped surrogate; all observed/tool tokens masked.

    Per-episode reduction is completed by the caller. Only sampled assistant
    action tokens, including actual end-of-turn tokens, enter this function.
    The k3 KL term is the conventional sampled nonnegative surrogate, not a
    claim of an exact full-vocabulary KL under off-policy minibatches.
    """
    import torch
    advantage, epsilon, beta = (finite(v, n) for v, n in
                               ((advantage, "advantage"), (epsilon, "epsilon"), (beta, "beta")))
    if not 0 < epsilon < 1 or beta < 0:
        raise ValueError("Invalid clipping or KL configuration")
    tensors = (new_logprobs, old_logprobs, reference_logprobs, policy_mask)
    if any(t.ndim != 1 or t.shape != new_logprobs.shape for t in tensors):
        raise ValueError("Log probabilities and policy mask must be matching flat tensors")
    if not torch.all((policy_mask == 0) | (policy_mask == 1)).item():
        raise ValueError("policy_mask must contain only 0 or 1")
    selected = policy_mask.bool()
    if not selected.any().item():
        raise ValueError("No policy-produced tokens in loss")
    new = new_logprobs[selected]
    old = old_logprobs.detach()[selected]
    ref = reference_logprobs.detach()[selected]
    if not all(torch.isfinite(v).all().item() for v in (new, old, ref)):
        raise ValueError("Non-finite selected log probabilities")
    log_ratio, log_ref_ratio = new - old, ref - new
    if (log_ratio.abs() > 60).any().item() or (log_ref_ratio.abs() > 60).any().item():
        raise ValueError("Unstable likelihood ratio; refusing silent numerical clipping")
    ratio = log_ratio.exp()
    surrogate = torch.minimum(ratio * advantage,
                              ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
    kl = torch.expm1(log_ref_ratio) - log_ref_ratio
    return (-surrogate + beta * kl).mean()

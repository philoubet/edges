"""
This module contains the Uncertainty class, which is responsible for handling
"""

import numpy as np
from scipy import stats
from edges.utils import safe_eval


def sample_cf_distribution(
    cf: dict,
    n: int,
    parameters: dict,
    random_state: np.random._generator.Generator,
    use_distributions: bool = True,
    SAFE_GLOBALS: dict = None,
) -> np.ndarray:
    """
    Generate n random CF values from the distribution info in the 'uncertainty' key.
    Falls back to a constant value if no uncertainty. If 'negative' == 1, samples are negated.
    """
    if not use_distributions or cf.get("uncertainty") is None:
        # If value is a string (expression), evaluate once
        value = cf["value"]
        if isinstance(value, str):
            value = safe_eval(
                expr=value,
                parameters=parameters,
                scenario_idx=0,
                SAFE_GLOBALS=SAFE_GLOBALS,
            )
        return np.full(n, value, dtype=float)

    unc = cf["uncertainty"]
    dist_name = unc["distribution"]
    params = unc["parameters"]
    negative = unc.get("negative", 0)

    if dist_name == "discrete_empirical":
        values = params["values"]
        weights = np.array(params["weights"])
        weights = weights / weights.sum() if weights.sum() != 0 else weights

        chosen_indices = random_state.choice(len(values), size=n, p=weights)

        samples = np.empty(n)

        for i, idx in enumerate(chosen_indices):
            item = values[idx]
            if isinstance(item, dict) and "distribution" in item:
                # Recursively sample this distribution
                samples[i] = sample_cf_distribution(
                    cf={"value": 0, "uncertainty": item},
                    n=1,
                    parameters=parameters,
                    random_state=random_state,
                    use_distributions=use_distributions,
                    SAFE_GLOBALS=SAFE_GLOBALS,
                )[0]
            else:
                samples[i] = item

    elif dist_name == "uniform":
        samples = random_state.uniform(params["minimum"], params["maximum"], size=n)

    elif dist_name == "triang":
        left = params["minimum"]
        mode = params["loc"]
        right = params["maximum"]
        samples = random_state.triangular(left, mode, right, size=n)

    elif dist_name == "normal":
        samples = random_state.normal(loc=params["loc"], scale=params["scale"], size=n)
        samples = np.clip(samples, params["minimum"], params["maximum"])

    elif dist_name == "lognorm":
        s = params["shape_a"]
        loc = params["loc"]
        scale = params["scale"]
        samples = stats.lognorm.rvs(
            s=s, loc=loc, scale=scale, size=n, random_state=random_state
        )
        samples = np.clip(samples, params["minimum"], params["maximum"])

    elif dist_name == "beta":
        a = params["shape_a"]
        b = params["shape_b"]
        x = random_state.beta(a, b, size=n)
        samples = params["loc"] + x * params["scale"]
        samples = np.clip(samples, params["minimum"], params["maximum"])

    elif dist_name == "gamma":
        samples = (
            random_state.gamma(params["shape_a"], params["scale"], size=n)
            + params["loc"]
        )
        samples = np.clip(samples, params["minimum"], params["maximum"])

    elif dist_name == "weibull_min":
        c = params["shape_a"]
        loc = params["loc"]
        scale = params["scale"]
        samples = stats.weibull_min.rvs(
            c=c, loc=loc, scale=scale, size=n, random_state=random_state
        )
        samples = np.clip(samples, params["minimum"], params["maximum"])

    else:
        samples = np.full(n, cf["value"], dtype=float)

    return -samples if negative else samples

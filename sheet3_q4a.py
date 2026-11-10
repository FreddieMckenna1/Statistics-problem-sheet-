"""
Problem Sheet 3, Question 4(a)

Four measured observables (x+, y+, x-, y-) with uncertainties and a correlation
matrix are assumed to be multivariate normally distributed. They are related to
the physics parameters (gamma, r, delta) by

    x± = r cos(delta ± gamma)
    y± = r sin(delta ± gamma)

We use the maximum likelihood (ML) method to estimate gamma, r and delta,
together with their uncertainties and correlations.

For a multivariate normal likelihood,

    -2 ln L = (x_obs - x_model)^T  V^-1  (x_obs - x_model)  + const

so maximising L is the same as minimising this chi-squared.
"""

import numpy as np
from iminuit import Minuit

# ---------------------------------------------------------------------------
# Measured values, uncertainties and correlation matrix
# Order: x+, y+, x-, y-
# ---------------------------------------------------------------------------
obs = np.array([-9.3, -1.3, 5.7, 6.5])
err = np.array([8.2, 8.4, 8.2, 8.3])

corr = np.array([
    [ 1.00, -0.10, -0.05,  0.10],
    [-0.10,  1.00,  0.10, -0.05],
    [-0.05,  0.10,  1.00,  0.10],
    [ 0.10, -0.05,  0.10,  1.00],
])

# Covariance matrix V_ij = rho_ij * sigma_i * sigma_j, and its inverse
cov = corr * np.outer(err, err)
cov_inv = np.linalg.inv(cov)


# ---------------------------------------------------------------------------
# Model and negative log-likelihood
# ---------------------------------------------------------------------------
def model(gamma, r, delta):
    """Predicted (x+, y+, x-, y-) for given parameters (angles in radians)."""
    return np.array([
        r * np.cos(delta + gamma),
        r * np.sin(delta + gamma),
        r * np.cos(delta - gamma),
        r * np.sin(delta - gamma),
    ])


def nll(gamma, r, delta):
    """Negative log-likelihood (up to a constant): 0.5 * chi^2."""
    d = obs - model(gamma, r, delta)
    return 0.5 * d @ cov_inv @ d


# ---------------------------------------------------------------------------
# Fit
# The trig functions make the likelihood multi-modal, so start the minimiser
# from a grid of starting points and keep the best (lowest NLL) result.
# ---------------------------------------------------------------------------
best = None
for g0 in np.linspace(0, np.pi, 7):
    for d0 in np.linspace(-np.pi, np.pi, 9):
        m = Minuit(nll, gamma=g0, r=5.0, delta=d0)
        m.errordef = Minuit.LIKELIHOOD  # 0.5, since we minimise -ln L
        m.limits["r"] = (0, None)       # r is a magnitude, so r >= 0
        m.migrad()
        if m.valid and (best is None or m.fval < best.fval):
            best = m

m = best
m.hesse()  # uncertainties from the second derivative of -ln L


# ---------------------------------------------------------------------------
# Results
# gamma and delta are only defined modulo 2*pi, and (gamma, delta) ->
# (gamma + pi, delta + pi) gives an identical prediction. We quote gamma in
# [0, 180) degrees and delta in (-180, 180] degrees.
# ---------------------------------------------------------------------------
gamma, r, delta = m.values["gamma"], m.values["r"], m.values["delta"]
if gamma % (2 * np.pi) >= np.pi:
    gamma, delta = gamma - np.pi, delta - np.pi
gamma = gamma % np.pi
delta = (delta + np.pi) % (2 * np.pi) - np.pi

s_gamma, s_r, s_delta = m.errors["gamma"], m.errors["r"], m.errors["delta"]
deg = 180 / np.pi

print("Maximum likelihood fit result")
print("-----------------------------")
print(f"  gamma = {gamma * deg:7.1f} ± {s_gamma * deg:5.1f} degrees")
print(f"  r     = {r:7.2f} ± {s_r:5.2f}")
print(f"  delta = {delta * deg:7.1f} ± {s_delta * deg:5.1f} degrees")
print(f"\n  -2 ln L at minimum (chi^2) = {2 * m.fval:.3f}")
print(f"  (4 observables - 3 parameters = 1 degree of freedom)")

print("\nCorrelation matrix (gamma, r, delta)")
names = ["gamma", "r", "delta"]
corr_fit = np.array(m.covariance.correlation())
print("          " + "".join(f"{n:>9}" for n in names))
for n, row in zip(names, corr_fit):
    print(f"  {n:<8}" + "".join(f"{v:9.3f}" for v in row))

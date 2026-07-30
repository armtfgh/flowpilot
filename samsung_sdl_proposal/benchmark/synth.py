"""Synthetic high-k thin-film landscape for benchmarking escalation policies.

Physics-INSPIRED, not physics-accurate: functional forms mimic literature-reported
qualitative behavior of HfO2/ZrO2-family ALD films (ALD temperature window,
impurity incorporation, anneal-driven crystallization, tetragonal-vs-monoclinic
phase competition, thickness-dependent tunneling). The purpose is to validate
*algorithms*, not to model any real material.

Structure of the problem
------------------------
Process inputs  u[0:5] (all normalized to [0,1]):
    u0  deposition temperature      (150-350 C)
    u1  Zr fraction (cycle ratio)   (0-1)
    u2  thickness                   (2-20 nm)
    u3  plasma power                (arb.)
    u4  purge time                  (arb.)
Anneal inputs   u[5:7]:
    u5  anneal temperature          (300-700 C)
    u6  anneal time                 (log scale, arb.)

Per-sample tool drift eta (unobservable from the recipe, but visible in the
structural fingerprint) perturbs impurity content and density.

Stage 1 (fast, cheap, every sample) -> structural fingerprint s:
    thickness, relative density, roughness, as-deposited crystallinity
Stage 2 (slow, expensive, escalated samples only) -> electrical outcome y:
    k (dielectric constant), log10 leakage current density

Hidden variable z ~ N(0,1): in the PARTIALLY crystallized regime (chi ~ 0.5,
mixed phase), leakage is dominated by percolation paths / local grain structure
that the fast observables cannot resolve -> irreducible (aleatoric) spread of
up to ~ +-1.5 decades. Everywhere else the mapping s -> y is tight.
"""
import numpy as np

# physical ranges for axis labelling
T_DEP = (150.0, 350.0)     # C
D_NM = (2.0, 20.0)         # nm
T_ANN = (300.0, 700.0)     # C

Z_COEF = 1.5               # decades of hidden-variable leakage spread at w_z = 1
MEAS_LOGJ = 0.15           # measurement noise, log10(A/cm^2)
MEAS_K = 0.5               # measurement noise on k
ETA_SD = 0.05              # tool-drift magnitude
HIDDEN_LABEL_WZ = 0.5      # w_z above this => "unpredictable" ground-truth label
LEARNABLE_WZ = 0.25        # w_z below this => region used for map-accuracy RMSE


def _sig(t):
    return 1.0 / (1.0 + np.exp(-t))


def structure(U, eta):
    """Deterministic structural state given process params and drift."""
    u0, u1, u2, u3, u4 = U[:, 0], U[:, 1], U[:, 2], U[:, 3], U[:, 4]
    d_nm = D_NM[0] + (D_NM[1] - D_NM[0]) * u2
    window = np.exp(-(((u0 - 0.55) / 0.28) ** 2))          # ALD temperature window
    c_imp = (0.55 * (1 - u0) + 0.45 * (1 - u4)) * (1 - 0.35 * u3)
    c_imp = np.clip(c_imp + eta, 0.0, 1.0)                  # drift raises impurities
    rho = np.clip(0.80 + 0.15 * window - 0.30 * c_imp - 0.45 * eta, 0.5, 1.0)
    phi0 = 0.5 * _sig(5 * (u0 - 0.75) + 2.5 * (u2 - 0.6) + 1.2 * (u1 - 0.5))
    sig_r = 0.15 + 0.5 * u2 + 0.8 * phi0 + 0.4 * (1 - rho)  # roughness, nm
    return dict(d_nm=d_nm, c_imp=c_imp, rho=rho, phi0=phi0, sig_r=sig_r)


def fingerprint(struct, rng=None):
    """Stage-1 measured structural fingerprint (4 features)."""
    s = np.stack([struct["d_nm"], struct["rho"], struct["sig_r"], struct["phi0"]], axis=1)
    if rng is not None:  # ~1% relative measurement noise
        s = s * (1 + 0.01 * rng.standard_normal(s.shape))
    return s


def outcome_mean(U, struct):
    """Deterministic part of the stage-2 outcome + hidden-variable weight w_z."""
    u1, u2, u5, u6 = U[:, 1], U[:, 2], U[:, 5], U[:, 6]
    c_imp, rho, phi0, d_nm = struct["c_imp"], struct["rho"], struct["phi0"], struct["d_nm"]

    drive = (1.20 * u5 + 0.35 * u2 + 0.50 * phi0
             + 0.15 * u6 - 0.15 * c_imp)
    chi = _sig(10.0 * (drive - 0.90))                        # crystallization degree
    phi_t = _sig(5.0 * (u1 - 0.42) - 3.2 * (u2 - 0.55))     # tetragonal fraction

    k_mean = 17.0 + 21.0 * chi * phi_t - 2.5 * chi * (1 - phi_t) - 5.0 * c_imp + 2.0 * rho
    logj_mean = (-8.2
                 + 5.5 * np.exp(-(d_nm - 2.0) / 3.0)        # direct tunneling
                 + 2.8 * c_imp                               # impurity-assisted
                 + 3.0 * (1.0 - rho)                         # porosity / defects
                 + 1.6 * chi * (1 - phi_t)                   # monoclinic grain boundaries
                 + 0.3 * chi * phi_t)
    w_z = 4.0 * chi * (1 - chi)                              # mixed-phase hidden regime
    return dict(k=k_mean, logj=logj_mean, w_z=w_z, chi=chi, phi_t=phi_t)


def sample_outcome(U, struct, rng):
    """One physical stage-2 measurement: hidden variable + measurement noise."""
    m = outcome_mean(U, struct)
    n = len(U)
    z = rng.standard_normal(n)
    logj = m["logj"] + Z_COEF * m["w_z"] * z + MEAS_LOGJ * rng.standard_normal(n)
    k = m["k"] + 1.0 * m["w_z"] * rng.standard_normal(n) + MEAS_K * rng.standard_normal(n)
    return k, logj


def make_set(n, rng, sobol=False, seed=0):
    """Generate n samples: process+anneal points, drift, fingerprint, truth."""
    if sobol:
        from scipy.stats import qmc
        U = qmc.Sobol(7, scramble=True, seed=seed).random(n)
    else:
        U = rng.random((n, 7))
    eta = ETA_SD * rng.standard_normal(n)
    struct = structure(U, eta)
    s = fingerprint(struct, rng)
    m = outcome_mean(U, struct)
    return dict(U=U, eta=eta, struct=struct, s=s,
                Ek=m["k"], Elogj=m["logj"], w_z=m["w_z"])


def features(dset, use_fingerprint=True):
    """Model inputs: (fingerprint, anneal) or (recipe, anneal) for the ablation."""
    if use_fingerprint:
        return np.hstack([dset["s"], dset["U"][:, 5:7]])
    return dset["U"]

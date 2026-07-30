"""Minimal ARD-RBF Gaussian process + most-likely heteroscedastic wrapper.

HeteroGP separates:
  epistemic variance  -- posterior variance of the latent function f
                         (shrinks with data; "sample more here")
  aleatoric variance  -- input-dependent noise learned from LOO residuals
                         (irreducible; "fast observables cannot resolve this")
Reference: Kersting et al., "Most Likely Heteroscedastic Gaussian Process
Regression" (ICML 2007) -- simplified two-pass variant.
"""
import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize

JITTER = 1e-8


class GP:
    """ARD-RBF GP. Noise: learned scalar, or fixed per-point variance vector."""

    def __init__(self, noise_var=None):
        self.fixed_noise = noise_var  # None -> learn scalar noise

    def _kern(self, A, B, ls, amp):
        d2 = ((A[:, None, :] - B[None, :, :]) / ls) ** 2
        return amp * np.exp(-0.5 * d2.sum(-1))

    def _nll(self, theta):
        ls = np.exp(theta[:self.dim])
        amp = np.exp(theta[self.dim])
        K = self._kern(self.X, self.X, ls, amp)
        if self.fixed_noise is None:
            K += np.exp(theta[self.dim + 1]) * np.eye(len(self.X))
        else:
            K += np.diag(self.fixed_noise + JITTER)
        K += JITTER * np.eye(len(self.X))
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        a = cho_solve((L, True), self.y)
        return float(0.5 * self.y @ a + np.log(np.diag(L)).sum())

    def fit(self, X, y, restarts=2, rng=None, min_ls=0.1, x0_extra=None):
        rng = rng or np.random.default_rng(0)
        self.X, self.y, self.dim = X, y, X.shape[1]
        n_par = self.dim + (1 if self.fixed_noise is None else 0) + 1
        best, best_nll = None, np.inf
        extra = ([np.asarray(x0_extra)] if x0_extra is not None
                 and len(x0_extra) == n_par else [])
        for r in range(restarts + len(extra)):
            if r >= restarts:  # warm start from a previous round's optimum
                x0 = extra[r - restarts].copy()
            else:
                x0 = np.zeros(n_par)
                x0[:self.dim] = np.log(max(1.0, min_ls)) if r == 0 else rng.uniform(
                    np.log(min_ls) + 0.2, 1.0, self.dim)
                x0[self.dim] = 0.0
                if self.fixed_noise is None:
                    x0[-1] = np.log(0.05)
            bounds = ([(np.log(min_ls), np.log(20))] * self.dim + [(np.log(0.05), np.log(20))]
                      + ([(np.log(1e-4), np.log(4.0))] if self.fixed_noise is None else []))
            res = minimize(self._nll, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 120})
            if res.fun < best_nll:
                best, best_nll = res.x, res.fun
        self.theta = best
        self.ls = np.exp(best[:self.dim])
        self.amp = np.exp(best[self.dim])
        self.noise = (np.exp(best[-1]) if self.fixed_noise is None
                      else None)
        K = self._kern(X, X, self.ls, self.amp)
        K += (self.noise * np.eye(len(X)) if self.fixed_noise is None
              else np.diag(self.fixed_noise))
        K += JITTER * np.eye(len(X))
        self.L = cholesky(K, lower=True)
        self.alpha = cho_solve((self.L, True), y)
        self.Kinv = cho_solve((self.L, True), np.eye(len(X)))
        return self

    def predict_f(self, Xs):
        """Posterior mean and EPISTEMIC variance of the latent f (no noise)."""
        Ks = self._kern(Xs, self.X, self.ls, self.amp)
        mu = Ks @ self.alpha
        v = cho_solve((self.L, True), Ks.T)
        var = np.maximum(self.amp - np.einsum("ij,ji->i", Ks, v), 1e-10)
        return mu, var

    def loo_residuals(self):
        """Closed-form leave-one-out residuals."""
        Kinv_y = self.Kinv @ self.y
        return Kinv_y / np.diag(self.Kinv)

    def loo_noise_targets(self):
        """Per-point noise-variance estimates with the epistemic part removed.

        E[r_loo_i^2] = var_loo_total_i = 1/[K^-1]_ii, which splits into the
        LOO epistemic variance plus the true noise at i. Subtracting the
        epistemic part de-biases the noise estimate in sparsely sampled areas.
        """
        r = self.loo_residuals()
        v_total = 1.0 / np.diag(self.Kinv)
        nv = (np.full(len(self.y), self.noise) if self.fixed_noise is None
              else self.fixed_noise)
        epi_loo = np.maximum(v_total - nv, 0.0)
        return np.log(np.maximum(r ** 2 - epi_loo, 1e-4))


class HeteroGP:
    """Two-pass most-likely heteroscedastic GP with epistemic/aleatoric split."""

    def __init__(self, n_iter=2):
        self.n_iter = n_iter

    def fit(self, X, y, rep_pairs=None, rng=None, warm=None):
        """rep_pairs: row-index pairs that are process replicates of the same
        recipe. Their paired difference is a pure, epistemic-free noise
        estimate -- these anchor the aleatoric field.
        warm: a previously fitted HeteroGP; its optima seed this fit
        (cumulative hyperparameter optimization across SDL rounds)."""
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-9
        self.ym, self.ys = y.mean(), y.std() + 1e-9
        Xn, yn = (X - self.xm) / self.xs, (y - self.ym) / self.ys
        rep_pairs = rep_pairs or []
        rep_rows = {i for p in rep_pairs for i in p}

        w_main = getattr(warm, "theta_main", None) if warm else None
        w_noise = getattr(warm, "theta_noise", None) if warm else None
        gp = GP().fit(Xn, yn, restarts=3, rng=rng, min_ls=0.5)
        for _ in range(self.n_iter):
            t_loo = gp.loo_noise_targets()
            keep = np.array([i not in rep_rows for i in range(len(yn))])
            Xt = [Xn[keep]]
            tt = [t_loo[keep]]
            for i, j in rep_pairs:  # unbiased anchors from replicate pairs
                est = np.log(0.5 * (yn[i] - yn[j]) ** 2 + 1e-4)
                Xt.append(Xn[[i]])
                tt.append([est])
            Xt, tt = np.vstack(Xt), np.concatenate(tt)
            # the noise field is a broad physical regime, not a point feature:
            # force it smooth so few points suffice to localize it
            self.noise_gp = GP().fit(Xt, tt, restarts=1, rng=rng, min_ls=0.5,
                                      x0_extra=w_noise)
            log_nv, _ = self.noise_gp.predict_f(Xn)
            nv = np.clip(np.exp(log_nv), 1e-4, 4.0)
            gp = GP(noise_var=nv).fit(Xn, yn, rng=rng, min_ls=0.5,
                                      x0_extra=w_main)
        self.gp = gp
        self.theta_main = gp.theta
        self.theta_noise = self.noise_gp.theta
        return self

    def predict(self, Xs):
        """Returns (mean, epistemic std, aleatoric std, noise-field epistemic
        variance) -- the last one in log-noise units, i.e. how uncertain the
        model still is about the LOCAL NOISE LEVEL at Xs."""
        Xn = (Xs - self.xm) / self.xs
        mu, var_epi = self.gp.predict_f(Xn)
        log_nv, var_lognv = self.noise_gp.predict_f(Xn)
        std_al = np.sqrt(np.clip(np.exp(log_nv), 1e-4, 4.0))
        return (mu * self.ys + self.ym,
                np.sqrt(var_epi) * self.ys,
                std_al * self.ys,
                var_lognv)

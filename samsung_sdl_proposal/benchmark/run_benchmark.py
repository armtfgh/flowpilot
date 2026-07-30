"""Compare stage-2 (electrical-test) escalation policies on the synthetic landscape.

Human-in-the-loop protocol: each round, the algorithm proposes ONE BATCH = one
working session of the human operator = 10 electrical measurements. A batch is
8 new samples + 2 process replicates (a duplicate sample grown with the same
recipe -- cheap in the ALD inner loop -- annealed and probed identically).
Replicate pairs give pure, epistemic-free noise estimates that anchor the
aleatoric field. Budget: 10 seed + 7 rounds x 10 = 80 measurements total,
identical for every policy.

Policies (all evaluated with the SAME heteroscedastic model on their own data;
only the acquisition rule differs, so the comparison isolates the policy):
  random     -- uniform choice among candidates
  sobol      -- space-filling in the full (process, anneal) space
  us_total   -- classic uncertainty sampling: max TOTAL predictive std
                (epistemic + aleatoric) -> gets trapped by irreducible noise
  epistemic  -- ours: information gain about the latent map (BALD),
                0.5*log(1 + s_epi^2 / s_al^2) -> learns what is learnable,
                flags the rest

Metrics per round:
  rmse_learn -- RMSE of predicted log10(J) vs true mean, in the learnable
                region (w_z < 0.25): map accuracy where a map exists
  auc        -- ROC-AUC of predicted aleatoric std as a detector of the
                ground-truth hidden-variable region (w_z > 0.5)
  frac_hidden-- fraction of unique escalated samples inside the hidden region
                (budget wasted on the unlearnable zone)
"""
import json
import time
import numpy as np
from sklearn.metrics import roc_auc_score

from synth import (make_set, features, sample_outcome,
                   LEARNABLE_WZ, HIDDEN_LABEL_WZ)
from gp import HeteroGP

N_CAND = 2500
N_TEST = 4096
BATCH_NEW = 8
BATCH_REP = 2
ROUNDS = 14
SEEDS = range(6)
POLICIES = ["random", "sobol", "us_total", "epistemic"]
MIN_DIST = 0.10  # batch-diversity radius in standardized feature space


def select_new(scores, Xn, blocked, n_pick):
    """Greedy top-score selection with a diversity radius inside the batch."""
    picked = []
    order = np.argsort(-scores)
    for i in order:
        if blocked[i] or not np.isfinite(scores[i]):
            continue
        if any(np.linalg.norm(Xn[i] - Xn[j]) < MIN_DIST for j in picked):
            continue
        picked.append(int(i))
        if len(picked) == n_pick:
            break
    k = 0
    while len(picked) < n_pick:
        i = int(order[k])
        if not blocked[i] and i not in picked:
            picked.append(i)
        k += 1
    return picked


def run_policy(policy, seed, on_round=None):
    rng = np.random.default_rng(1000 + seed)
    meas_rng = np.random.default_rng(9000 + seed)
    cand = make_set(N_CAND, rng)
    test = make_set(N_TEST, rng, sobol=True, seed=seed)

    Xc = features(cand)
    Xt = features(test)
    Xcn = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)

    learn_mask = test["w_z"] < LEARNABLE_WZ
    hid_label = (test["w_z"] > HIDDEN_LABEL_WZ).astype(int)

    def measure(i):
        """One physical stage-2 run of candidate i (fresh hidden variable)."""
        U1 = cand["U"][[i]]
        st1 = {k: v[[i]] for k, v in cand["struct"].items()}
        _, logj = sample_outcome(U1, st1, meas_rng)
        return float(logj[0])

    rows, y_rows, rep_pairs = [], [], []
    chosen = np.zeros(N_CAND, bool)

    def add_batch(new_idx):
        """8 new + replicate the 2 first (highest-priority) picks."""
        for i in new_idx:
            chosen[i] = True
            rows.append(i)
            y_rows.append(measure(i))
        for i in new_idx[:BATCH_REP]:
            rows.append(i)
            y_rows.append(measure(i))
            first = rows.index(i)
            rep_pairs.append((first, len(rows) - 1))

    # seed batch: 8 random new + 2 replicates = 10 measurements
    add_batch(list(rng.choice(N_CAND, BATCH_NEW, replace=False)))

    if policy == "sobol":
        from scipy.stats import qmc
        sob = qmc.Sobol(7, scramble=True, seed=seed).random(
            BATCH_NEW * ROUNDS + 64)
        sob_ptr = 0

    hist = {"n": [], "rmse_learn": [], "rmse_all": [], "auc": [],
            "frac_hidden": [], "queries": None}

    model = None
    for rnd in range(ROUNDS + 1):
        model = HeteroGP().fit(Xc[rows], np.array(y_rows),
                               rep_pairs=rep_pairs,
                               rng=np.random.default_rng(seed),
                               warm=model)
        if on_round is not None:
            on_round(len(rows), model, list(rows), cand)
        mu, _, sal, _ = model.predict(Xt)
        res = mu - test["Elogj"]
        uniq = sorted(set(rows))
        hist["n"].append(len(rows))
        hist["rmse_learn"].append(float(np.sqrt((res[learn_mask] ** 2).mean())))
        hist["rmse_all"].append(float(np.sqrt((res ** 2).mean())))
        hist["auc"].append(float(roc_auc_score(hid_label, sal)))
        hist["frac_hidden"].append(
            float((cand["w_z"][uniq] > HIDDEN_LABEL_WZ).mean()))

        if rnd == ROUNDS:
            break

        if policy == "random":
            free = np.where(~chosen)[0]
            new_idx = list(rng.choice(free, BATCH_NEW, replace=False))
        elif policy == "sobol":
            new_idx = []
            while len(new_idx) < BATCH_NEW:
                p = sob[sob_ptr]; sob_ptr += 1
                d = np.linalg.norm(cand["U"] - p[None], axis=1)
                d[chosen] = np.inf
                for i in new_idx:
                    d[i] = np.inf
                new_idx.append(int(np.argmin(d)))
        else:
            _, sepi, sal_c, nvar = model.predict(Xc)
            if policy == "us_total":
                score = np.sqrt(sepi ** 2 + sal_c ** 2)
            else:
                # ours: joint information gain (nats) about (a) the latent map
                # -- BALD, an observation drowned in irreducible noise teaches
                # nothing about f -- and (b) the NOISE FIELD itself, whose
                # observation noise from a single replicate pair is
                # var[log chi^2_1] = pi^2/2
                gain_f = 0.5 * np.log1p(sepi ** 2 / sal_c ** 2)
                gain_noise = 0.5 * np.log1p(nvar / (np.pi ** 2 / 2))
                score = gain_f + gain_noise
            score = score.copy()
            score[chosen] = -np.inf
            n_greedy = BATCH_NEW - (1 if policy == "epistemic" else 0)
            new_idx = select_new(score, Xcn, chosen, n_greedy)
            if policy == "epistemic":
                # epsilon-exploration safeguard: one random pick per batch
                # prevents lock-in on a confidently wrong noise map (a wrongly
                # "noisy" region would otherwise never be re-visited/corrected)
                free = np.where(~chosen)[0]
                free = [i for i in free if i not in new_idx]
                new_idx.append(int(rng.choice(free)))
                # replicate the picks that most reduce noise-field uncertainty
                order = np.argsort(-nvar[new_idx])
                new_idx = [new_idx[k] for k in order]

        add_batch(new_idx)

    uniq = sorted(set(rows))
    hist["queries"] = {
        "d_nm": cand["struct"]["d_nm"][uniq].tolist(),
        "t_ann": (300 + 400 * cand["U"][uniq, 5]).tolist(),
        "w_z": cand["w_z"][uniq].tolist(),
    }
    return hist


def main():
    t0 = time.time()
    out = {}
    for policy in POLICIES:
        out[policy] = []
        for seed in SEEDS:
            h = run_policy(policy, seed)
            out[policy].append(h)
            print(f"{policy:10s} seed {seed}: "
                  f"rmse_learn {h['rmse_learn'][0]:.3f} -> {h['rmse_learn'][-1]:.3f}  "
                  f"auc -> {h['auc'][-1]:.3f}  "
                  f"frac_hidden {h['frac_hidden'][-1]:.2f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    with open("results.json", "w") as f:
        json.dump(out, f)
    print(f"done in {time.time()-t0:.0f}s -> results.json")


if __name__ == "__main__":
    main()

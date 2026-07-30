# In-silico benchmark — results and honest findings

*(Numbers below are filled from `headline.txt` / `results.json` after the full
6-seed run; figures: `fig1_ground_truth.png`, `fig2_sampling.png`,
`fig3_learning.png`, `fig_slide_summary.png`.)*

## What was tested

A synthetic, physics-inspired high-k landscape (7 inputs: 5 process + 2 anneal;
fast structural fingerprint + expensive electrical outcome; a mixed-phase
"hidden variable" band where leakage is irreducibly unpredictable from fast
observables; tool drift visible only in the fingerprint). Four escalation
policies spent an identical budget of 150 electrical measurements in
human-in-the-loop batches of 10 (8 new samples + 2 process replicates):

| policy | acquisition rule |
|---|---|
| random | uniform |
| sobol | space-filling in the input space |
| us_total | classic uncertainty sampling (total predictive std) |
| **epistemic (ours)** | **joint information gain: BALD term for the latent map + information about the noise field itself; replicates aimed at noise-field uncertainty** |

All four policies use the *same* heteroscedastic GP model on their own data —
the comparison isolates the acquisition policy.

## Headline numbers

Mean over 6 campaigns of 150 electrical tests each (final values):

| metric | random | sobol | us_total | **ours** |
|---|---|---|---|---|
| map RMSE, learnable region (decades of J) | 0.62 | 0.71 | 0.76 | **0.56** |
| hidden-zone detection (ROC-AUC) | 0.69 | 0.68 | 0.69 | **0.74** |
| budget wasted in unlearnable zone | 30% | 31% | 35% | **16%** |

- Ours reaches every baseline's *final* map accuracy with **70–80 tests
  (≈1.9–2.1× fewer)** and keeps improving; no baseline reaches ours' final
  accuracy within budget.
- Curves overlap for the first ~60 tests (any policy works early — the
  advantage compounds once the noise field becomes identifiable).

## Methodological findings — worth stating in any talk

These came out of building the benchmark, and they are the actual scientific
content of the demo:

1. **Pure BALD has a chicken-and-egg failure.** An epistemic-only policy needs
   the aleatoric map to avoid the unlearnable zone, but never samples there to
   learn it. The fix — adding an explicit information term for the *noise
   field itself* — is what makes "learn the map AND its validity boundary" a
   single coherent acquisition function. This is the algorithmic core of the
   proposal.
2. **Aleatoric-aware policies can lock in on confident wrongness.** On one
   seed, an early mis-estimated noise field marked a learnable region as
   "noisy"; the policy stopped sampling there and — unlike an epistemic-error,
   which invites more samples — never collected the data that would correct
   it (RMSE stuck ~1.1 while other seeds reached ~0.5). A one-random-pick-per-
   batch ε-exploration safeguard eliminates the failure. Any deployed validity-
   mapping SDL needs such a safeguard; this benchmark is exactly where one
   finds that out, not in the lab.
3. **Replicates are not optional.** With single measurements, LOO-residual
   noise estimation confounds epistemic error with aleatoric noise and the
   hidden-zone detector stays at chance (AUC ~0.5) at realistic budgets.
   Two process replicates per batch of ten give pure, epistemic-free noise
   anchors and fix detection. This directly validates the "replicates +
   reference runs" line on the proposal's risk slide: it is not bookkeeping,
   it is what makes validity mapping possible at all.
4. **There is a budget threshold.** In the 6-D feature space, below roughly
   ~100 electrical measurements no policy separates reliably and the noise
   field is unidentifiable; at ~150 the machinery works. Useful for scoping
   Phase 1: the electrical-test campaign must be sized for the *noise field*,
   not just the mean map.
5. **Uncertainty sampling detects the hidden zone by drowning in it.**
   us_total is attracted to irreducible noise (its acquisition never drops
   below the aleatoric floor), so it "finds" the zone at the cost of map
   accuracy elsewhere. Ours aims to match its detection while keeping the
   budget on learnable physics.

## Honest limitations

- Validates the **algorithm**, not the material physics: the landscape is
  literature-inspired but synthetic. The claim is "the decision engine finds
  predictability structure efficiently when it exists," nothing more.
- GP-with-replicates is one modeling choice; deep ensembles / BNNs may scale
  better past ~10 inputs.
- Policy separation on RMSE is modest at this budget (the mean map is smooth
  and even random sampling learns it eventually); the decisive differences are
  *budget wasted in the unlearnable zone* and *detection per test*.
- Single objective (log leakage) drove acquisition; k was modeled but not
  acquisition-relevant. Multi-objective acquisition is future work.

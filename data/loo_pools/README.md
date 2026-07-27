# LOO Probing Pools (M, C, B) — Extracted from Notebook Source

This folder contains the **three per-sequence pools** used by the leave-one-out
(LOO) cross-sequence probing protocol in the paper *Probe-Geometry Alignment:
Erasing the Cross-Sequence Memorization Signature Below Chance*.

These pools were previously defined only inline inside the `probe_core` cell
at the top of each main LOO notebook
(`notebooks/02_pythia.ipynb` cell 129, `notebooks/03_gpt-2-medium.ipynb`
cell 100, and `notebooks/04_mistral-7b.ipynb` cells 16 and 21). They are
serialized here verbatim for reproducibility.

## File layout

```
loo_pools/
├── README.md                  this file
├── NEUTRAL_CONTEXTS.json      24 short neutral prefixes shared across all models
├── pythia70m/
│   ├── POOL_M.json            9 candidate memorized sequences, keyed by string ID
│   ├── POOL_C.json            9 register-matched paraphrased controls (C class)
│   ├── POOL_B.json            9 further-paraphrased controls (B class)
│   └── valid_keys.json        7-key subset of POOL_M that passes log P/tok screening
├── gpt2medium/                identical POOL_M/C/B content as pythia70m (shared probe_core); same 7-key valid_keys
│   └── ...
└── mistral7b/
    ├── POOL_M.json            5 memorized sequences (a smaller, separately defined set)
    ├── POOL_C.json            5 register-matched paraphrased controls
    ├── POOL_B.json            5 further-paraphrased controls
    └── valid_keys.json        all 5 keys (Mistral-7B uses the pre-screened set)
```

Each `POOL_*.json` file is a JSON object whose **keys are the string IDs**
(`mit_license`, `apache_license`, `gpl_header`, `bsd_license`, `python_main`,
`creative_commons`, `lorem_ipsum`, plus `shebang_python` and `wikipedia_python`
for the candidate set on Pythia/GPT-2) and whose **values are the full verbatim
strings** used by the probe.

Pairing is by key: for a given model and key `k`, the memorized text is
`POOL_M[k]`, the matched control is `POOL_C[k]`, and the second-class control
used by the pure-distinguishability baseline is `POOL_B[k]`.

## How these pools enter the LOO computation

For each model, the procedure (see Appendix on LOO protocol in the paper) is:

1. **Screen** each candidate key against the base model by log probability per
   token. A key passes if `log P/tok(POOL_M[k]) > theta_M`,
   `log P/tok(POOL_C[k]) < theta_C`, and `log P/tok(POOL_B[k]) < theta_C` (see
   `screen_pools()` in the `probe_core` cell). Per-model thresholds used in the
   paper: Pythia-70M `(theta_M, theta_C) = (-2.0, -3.0)` (notebook 02 cell 132),
   GPT-2 Medium `(-1.5, -2.5)` (notebook 03 cell 103), Mistral-7B uses the
   pre-screened set from `E1_results.json` and additionally re-screens POOL_B
   against `log P/tok < -2.5` (notebook 04 cell 21). The resulting subset is
   `valid_keys.json`. For Pythia-70M and GPT-2 Medium this drops the candidate
   set from 9 to 7 (the same 7 keys for both, since they share `probe_core`);
   for Mistral-7B the inline `POOL_M` already contains the 5 pre-validated
   pairs.
2. **Extract activations** for each `(k, b)` with `k` in `valid_keys` and
   `b` in `NEUTRAL_CONTEXTS`, by prepending `b` to each pool entry and reading
   the last-token residual stream at every depth.
3. **True LOO**: for each held-out `k_j` in `valid_keys`, train a logistic
   regression on `POOL_M` vs `POOL_C` activations for `valid_keys \ {k_j}`,
   evaluate on the `(k_j, b)` activations. Average across held-out keys per
   depth.
4. **Pure-distinguishability LOO**: identical procedure but with `POOL_C`
   activations as the positive class and `POOL_B` activations as the negative
   class — i.e. two non-memorized variants of the same key, screened to fail
   the memorization threshold. This isolates string-separability structure
   from memorization-specific structure.
5. **Memorization-specific gap**: `mean(True LOO) - mean(Pure LOO)` per depth.

In notebook code, step 4 is realised by `reps_pure = {k: {'mem': reps[k]['ctl'],
'ctl': reps[k]['B']} for k in valid_keys}` and then `loo_accuracy(reps_pure,
valid_keys, d)`. See `02_pythia.ipynb` cell 134 for a worked example.

## Relationship to `data/*.json`

The files `data/pythia_memorized.json`, `data/pythia_clean.json`,
`data/mistral_memorized.json`, `data/mistral_clean.json`, and `data/heldout.json`
are used by the downstream MLDU-E / PGA-erasure pipeline notebooks
(`mldu-e-*.ipynb`, `MLDU_E_mistral_kaggle_pipeline2.ipynb`), **not** by the main
LOO probing notebooks `02_pythia.ipynb`, `03_gpt-2-medium.ipynb`, or
`04_mistral-7b.ipynb`. They are constructed differently (a memorized vs.
unrelated-prose comparison rather than the three-pool M/C/B setup) and should
not be confused with `POOL_M / POOL_C / POOL_B` here.

## Provenance

All strings in this folder were extracted via `ast.literal_eval` directly from
the assignment expressions in the notebook source cells. No content was
modified, paraphrased, or regenerated.
